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

print("--- [DEBUG] Checking environment keys ---")
print("Gemini Key Found:", os.getenv("GEMINI_API_KEY") is not None)
print("----------------------------------------")

class RagService:
    def __init__(self, retrieval_service: RetrievalService, dataset: Any) -> None:
        self.retrieval: RetrievalService = retrieval_service
        self.dataset = dataset
        

        print("[RAG] جاري الاتصال بمخزن الوثائق (DocStore)...")
        self.docs_store = dataset.docs_store()

        self.gemini_key: str | None = os.getenv("GEMINI_API_KEY")
        
        self.provider: str | None = None
        self.client: Any = None
        self.model_name: str = ""

        if self.gemini_key and genai is not None:

            print("[RAG] تم رصد مفتاح Gemini. جاري تهيئة العميل الخاص بـ Google GenAI...")
            self.client = genai.Client(api_key=self.gemini_key)
            self.model_name = "gemini-2.5-flash"
            self.provider = "gemini"
        else:
            print("[RAG WARNING] مفتاح Gemini مفقود أو المكتبة غير مثبتة. تفعيل وضع الطوارئ المحلي.")

    def _smart_truncate(self, text: str, max_chars: int = 1500) -> str:

        if len(text) <= max_chars:
            return text
        truncated: str = text[:max_chars]
        last_period: int = truncated.rfind('. ')
        if last_period != -1:
            return truncated[:last_period + 1] + " ... [Truncated]"
        return truncated + " ... [Truncated]"

    def ask_rag(self, user_question: str, top_k: int = 3, alpha: float = 0.7) -> Dict[str, Any]:
        print(f"\n[RAG] 1. جاري استرجاع أفضل {top_k} وثائق مرتبطة بالسؤال عبر المحرك الهجين...")
        
        query_tokens: List[str] = preprocess(user_question)
        
        retrieved_docs = self.retrieval.search_hybrid_parallel(
            query_text=user_question, 
            query_tokens=query_tokens, 
            alpha=alpha, 
            top_k=top_k
        )
        
        if not retrieved_docs:
            return {
                "answer": "Insufficient data in the retrieved medical research to answer this query accurately.",
                "referenced_docs": []
            }

        print("[RAG] 2. جاري دمج النصوص المسترجعة لبناء السياق الحاكم...")
        context_parts: List[str] = []
        seen_docs: Set[str] = set() #  منع تكرار الوثائق لتقليل استهلاك الرموز
        
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
You are a precise Medical Research Synthesizer (Gemini 2.5). Your task is strictly Retrieval-Augmented Generation (RAG).
</system_instruction>

<rules>
1. **ZERO TOLERANCE FOR HALLUCINATION**: Before typing any answer, explicitly check if the "Retrieved Context" contains the exact answer to the "User Inquiry". If the context is missing the specific answer, missing statistical data, or requires external medical knowledge—OUTPUT ONLY the exact phrase: "Insufficient data in the retrieved medical research to answer this query accurately." DO NOT add extra sentences.
2. **MANDATORY CITATIONS**: Every claim, number, or result MUST be followed by the source document ID strictly formatted as [Doc: doc_id]. If multiple sources support the claim, format as [Doc: id1, id2].
3. **CONFLICT RESOLUTION**: If the context contains conflicting studies or results, present both objectively in a "Contradictions" section. Do not pick a side unless the context explicitly declares a consensus.
4. **FORMATTING & LENGTH**: Output must be in valid Markdown. Strictly limit the response to a maximum of 250 words. Be concise.
5. **SAFETY**: If the user inquiry is NSFW, offensive, or non-medical, output exactly: "Query rejected. This system strictly processes academic and medical inquiries."
</rules>

<context>
{context}
</context>

<user_query>
{user_question}
</user_query>

<output_schema>
If the data is sufficient, structure your reply exactly as follows:

**Summary:** (One-sentence direct answer to the user)
**Key Evidence:**
- (Point 1) [Doc: X]
- (Point 2) [Doc: Y]
**Contradictions (if any):** (Briefly mention conflicts)
</output_schema>

<process>
Step 1: Verify if the context answers the query.
Step 2: If NO -> output "Insufficient data...". If YES -> extract facts and map them to Doc IDs.
Step 3: Generate the Markdown response strictly following the Output Schema.
</process>

=== ACADEMIC ANALYSIS ===
"""

        print(f"[RAG] 3. جاري إرسال البيانات عبر بوابة ({self.provider}) لتوليد الإجابة النهائية...")
        
        if not self.client or not self.provider:
            return self._fallback_response(retrieved_docs)
            
        try:
            if self.provider == "gemini":
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
            print(f"[RAG WARNING] تعذر الاتصال بالمزود المحدد ({str(e)}). تفعيل خطة الطوارئ المحلية...")
            return self._fallback_response(retrieved_docs)

    def _fallback_response(self, retrieved_docs: List[Tuple[str, float]]) -> Dict[str, Any]:
        fallback_answer: str = "System Note: AI Generation Service is currently unavailable. Displaying the most relevant localized research text segment as a fail-safe backup:\n\n"
        
        if retrieved_docs:
            first_doc_id: str = retrieved_docs[0][0]
            first_doc = self.docs_store.get(first_doc_id)
            
            if first_doc:
                doc_text: str = extract_text(first_doc)
                snippet: str = self._smart_truncate(doc_text, max_chars=800)
                fallback_answer += f'"{snippet}" [Direct Source Backup Doc: {first_doc_id}]'
        
        return {
            "answer": fallback_answer,
            "referenced_docs": [doc_id for doc_id, _ in retrieved_docs]
        }