import os
from dotenv import load_dotenv
from google import genai
from services.retrieval_service import RetrievalService
from services.preprocessing_service import extract_text, preprocess
from typing import Dict, Any, List, Tuple

load_dotenv()

class RagService:
    def __init__(self, retrieval_service: RetrievalService, dataset: Any) -> None:
        self.retrieval: RetrievalService = retrieval_service
        self.dataset = dataset
        
        print("[RAG] جاري الاتصال بمخزن الوثائق (DocStore)...")
        self.docs_store = dataset.docs_store()

        api_key: str | None = os.getenv("GEMINI_API_KEY")
       
        if not api_key:
            print("[RAG WARNING] مفتاح GEMINI_API_KEY مفقود. سيعمل النظام في وضع الطوارئ.")
            self.client = None
        else:
            self.client = genai.Client(api_key=api_key)
            
        self.model_name: str = "gemini-2.5-flash"

    def ask_rag(self, user_question: str, top_k: int = 3) -> Dict[str, Any]:

        print(f"\n[RAG] 1. جاري استرجاع أفضل {top_k} وثائق مرتبطة بالسؤال عبر المحرك الهجين...")
        
        query_tokens = preprocess(user_question)
        retrieved_docs = self.retrieval.search_hybrid_parallel(
            query_text=user_question, 
            query_tokens=query_tokens, 
            alpha=0.7, # إعطاء وزن أعلى للبحث الدلالي لأنه سؤال
            top_k=top_k
        )
        
        if not retrieved_docs:
            return {
                "answer": "عذراً، لم يتم العثور على أي وثائق ملائمة للإجابة على سؤالك.",
                "referenced_docs": []
            }

        print("[RAG] 2. جاري دمج النصوص المسترجعة لبناء السياق الحاكم...")
        context_parts: List[str] = []
        for doc_id, score in retrieved_docs:
            doc = self.docs_store.get(doc_id)
            if doc:
                doc_text: str = extract_text(doc)

                snippet = doc_text[:1500] + "..." if len(doc_text) > 1500 else doc_text
                context_parts.append(f"[Document ID: {doc_id}]\nContent: {snippet}")
                
        context: str = "\n\n".join(context_parts)

        prompt: str = f"""You are an elite academic research assistant and medical data analyst. 
Your primary task is to answer the user's question with absolute precision, relying EXCLUSIVELY on the provided "Retrieved Documents Context".

[STRICT RULES]
1. ZERO HALLUCINATION: If the answer is not explicitly contained within the provided context, you MUST reply strictly with: "لا توجد معلومات كافية في الأبحاث للإجابة على هذا السؤال."
2. DIRECT CITATION: Every factual claim, statistic, or medical finding in your answer MUST be immediately followed by the specific document ID it came from, formatted exactly as [Doc: doc_id].
3. LANGUAGE: You MUST answer in clear, professional Arabic.
4. STRUCTURED OUTPUT: Format your response using clear Markdown (e.g., bullet points, bold text for key terms, and short paragraphs) to ensure readability on mobile applications.

Retrieved Documents Context:
{context}

User Question: {user_question}

Answer in Arabic:"""

        print("[RAG] 3. جاري إرسال البيانات للنموذج اللغوي لتوليد الإجابة النهائية...")
        
        if not self.client:
            return self._fallback_response(retrieved_docs)
            
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            return {
                "answer": response.text,
                "referenced_docs": [doc_id for doc_id, _ in retrieved_docs]
            }
        except Exception as e:
            print(f"[RAG WARNING] تعذر الاتصال بـ Gemini ({str(e)}). تفعيل خطة الطوارئ...")
            return self._fallback_response(retrieved_docs)

    def _fallback_response(self, retrieved_docs: List[Tuple[str, float]]) -> Dict[str, Any]:
        """شرح الكود باللغة العربية: دالة مساعدة لتوليد إجابة الطوارئ"""
        fallback_answer = "عذراً، خوادم الذكاء الاصطناعي غير متاحة حالياً. كبديل فوري، إليك أهم مقتطف من الوثيقة الأولى ذات الصلة:\n\n"
        
        first_doc_id = retrieved_docs[0][0]
        first_doc = self.docs_store.get(first_doc_id)
        
        if first_doc:
            doc_text = extract_text(first_doc)
            fallback_answer += f'"{doc_text[:600]}..."'
        
        return {
            "answer": fallback_answer,
            "referenced_docs": [doc_id for doc_id, _ in retrieved_docs]
        }