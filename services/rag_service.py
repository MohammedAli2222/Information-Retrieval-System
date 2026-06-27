import os
from dotenv import load_dotenv
from typing import Dict, Any, List, Tuple, Set

try:
    from google import genai
except ImportError:
    genai = None

from services.retrieval_service import RetrievalService
from services.preprocessing_service import extract_text, preprocess

load_dotenv()

class RagService:
    def __init__(self, retrieval_service: RetrievalService, dataset: Any) -> None:
        self.retrieval: RetrievalService = retrieval_service
        self.dataset = dataset
        

        self.docs_store = dataset.docs_store()

        self.gemini_key: str | None = os.getenv("GEMINI_API_KEY")
        self.provider: str | None = None
        self.client: Any = None
        self.model_name: str = ""

        if self.gemini_key and genai is not None:
            self.client = genai.Client(api_key=self.gemini_key)
            self.model_name = "gemini-2.5-flash"
            self.provider = "gemini"

    def _smart_truncate(self, text: str, max_chars: int = 1500) -> str:

        if len(text) <= max_chars:
            return text
        truncated: str = text[:max_chars]
        last_period: int = truncated.rfind('. ')
        if last_period != -1:
            return truncated[:last_period + 1] + " ... [Truncated]"
        return truncated + " ... [Truncated]"

    def ask_rag(self, user_question: str, top_k: int = 3, alpha: float = 0.7) -> Dict[str, Any]:

        query_tokens: List[str] = preprocess(user_question)
        
        retrieved_docs = self.retrieval.search_hybrid_parallel(
            query_text=user_question, 
            query_tokens=query_tokens, 
            alpha=alpha, 
            top_k=top_k
        )
        
        if not retrieved_docs:
             return {
                "answer": "لا توجد وثائق مسترجعة في قاعدة البيانات مطابقة لهذا الاستعلام.",
                "referenced_docs": []
            }


        context_parts: List[str] = []
        seen_docs: Set[str] = set() 
        
        for doc_id, score in retrieved_docs:
            if doc_id in seen_docs:
                continue
                
            doc = self.docs_store.get(doc_id)
            if doc:
                doc_text: str = extract_text(doc)
                snippet: str = self._smart_truncate(doc_text)
                context_parts.append(f"[Document ID: {doc_id}]\nContent: {snippet}")
                seen_docs.add(doc_id)
                
            context: str = "\n\n".join(context_parts)


        prompt: str = f"""<system_instruction>
You are a Lead Clinical Data Scientist and Medical Researcher. Your task is to perform highly advanced Retrieval-Augmented Generation (RAG) by synthesizing complex epidemiological and medical research.
</system_instruction>

<rules>
1. **ADVANCED SYNTHESIS (The Core Directive)**: Do not simply list facts. Read the provided <context>, deeply analyze the semantic connections, and weave them into a sophisticated, flowing academic narrative. Use transitional phrases, contrast differing viewpoints, and highlight consensus.
2. **EVIDENCE WEIGHING**: If the context contains multiple studies with varying results, act as an expert reviewer. Compare them objectively (e.g., "While one study suggests X [Doc: id], broader analyses indicate Y [Doc: id]").
3. **TONE & STYLE**: Adopt the tone of a high-level medical journal (like The Lancet or NEJM). The output must be engaging, highly readable, and professionally structured. Avoid robotic phrasing like "The context states...".
4. **GRACEFUL HANDLING OF GAPS**: If the exact data point (like a specific percentage or date) is missing, seamlessly pivot to what IS known about the broader mechanism or related findings without breaking the academic flow.
5. **FLUID CITATIONS**: Embed citations seamlessly into the prose using brackets immediately after the scientific claim, exactly like this: [Doc: xyz123].
6. **FORMATTING**: Use Markdown strategically. Use bolding for key medical terms and bullet points ONLY if summarizing a complex list of symptoms or outcomes. 
7. **ABSOLUTE GROUNDING**: Despite the advanced, natural tone, 100% of the medical claims MUST originate from the <context>. No external hallucination is permitted.
</rules>

<context>
{context}
</context>

<user_query>
{user_question}
</user_query>
"""

        if not self.client or not self.provider:
            return self._fallback_response(retrieved_docs)
            
        try:

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            answer_text: str = response.text
            
            return {
                "answer": answer_text,
                "referenced_docs": [doc_id for doc_id, _ in retrieved_docs]
            }
            
        except Exception as e:
            return self._fallback_response(retrieved_docs)

    def _fallback_response(self, retrieved_docs: List[Tuple[str, float]]) -> Dict[str, Any]:
        # شرح الكود باللغة العربية: خطة الطوارئ في حال فشل الاتصال بالنموذج
        fallback_answer: str = "System Note: AI Generation Service is currently unavailable. Displaying the top relevant context:\n\n"
        
        if retrieved_docs:
            first_doc_id: str = retrieved_docs[0][0]
            first_doc = self.docs_store.get(first_doc_id)
            
            if first_doc:
                doc_text: str = extract_text(first_doc)
                snippet: str = self._smart_truncate(doc_text, max_chars=800)
                fallback_answer += f'"{snippet}" [Doc: {first_doc_id}]'
        
        return {
            "answer": fallback_answer,
            "referenced_docs": [doc_id for doc_id, _ in retrieved_docs]
        }