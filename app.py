import streamlit as st
import requests
from typing import Dict, Any, Optional, List

# إعداد واجهة الصفحة
st.set_page_config(page_title="Bio-IR Engine | CORD-19", layout="wide", page_icon="🧬")

API_BASE_URL: str = "http://127.0.0.1:8000"


def init_session_state() -> None:
    # تهيئة المتغيرات في ذاكرة الجلسة لمنع اختفاء البيانات عند إعادة التحميل
    if "search_results" not in st.session_state:
        st.session_state.search_results = None
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

# ==========================================
# 🏥 الحقن المخصص للستايل (Medical Glassmorphism CSS)
# ==========================================
def inject_custom_css() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;800&display=swap');

    /* الخطوط العامة */
    html, body {
        font-family: 'Cairo', sans-serif;
    }
    
    /* تطبيق الخط المخصص على النصوص فقط لحماية أيقونات Streamlit */
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdownContainer, input, button {
        font-family: 'Cairo', sans-serif;
    }

    /* إصلاح مشكلة الأيقونات (مثل زر القائمة الجانبية keyboard_double) */
    [data-testid="stSidebarCollapsedControl"] span, 
    [data-testid="stHeader"] span,
    .material-symbols-rounded {
        font-family: "Material Symbols Rounded", "Material Icons", sans-serif !important;
    }
    
    /* الخلفية الرئيسية - صورة فيروسات تحت الميكروسكوب هادئة جداً */
    .stApp {
        background: url('https://images.unsplash.com/photo-1584036561566-baf8f5f1b144?auto=format&fit=crop&q=80&w=2000') no-repeat center center fixed;
        background-size: cover;
    }
    
    /* إضافة طبقة داكنة فوق الصورة لتهدئة العين (Overlay) */
    .stApp::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background-color: rgba(10, 15, 25, 0.85); /* شفافية داكنة مريحة */
        z-index: 0;
    }
    
    /* التأكد من أن المحتوى يظهر فوق الطبقة الداكنة */
    .block-container {
        position: relative;
        z-index: 1;
    }

    /* الشريط الجانبي بتأثير الزجاج الطبي */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.6) !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* التبويبات (Tabs) - تصميم نظيف واحترافي */
    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 5px;
        backdrop-filter: blur(5px);
    }
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(59, 130, 246, 0.2) !important; /* لون أزرق طبي */
        color: #60a5fa !important;
        border-bottom: 2px solid #3b82f6 !important;
    }

    /* حقول الإدخال */
    .stTextInput>div>div>input, .stChatInput>div>div>input {
        background-color: rgba(15, 23, 42, 0.7);
        color: #e2e8f0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 12px;
        backdrop-filter: blur(5px);
    }
    .stTextInput>div>div>input:focus, .stChatInput>div>div>input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3);
    }

    /* العناوين الطبية */
    .virus-title {
        color: #f8fafc;
        font-size: 2.8em;
        font-weight: 800;
        margin-bottom: 5px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.5);
    }
    .sub-title {
        color: #94a3b8;
        font-size: 1.2em;
        margin-bottom: 30px;
        font-weight: 400;
    }
    
    /* البطاقات (Containers) - شفافية زجاجية أنيقة */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(30, 41, 59, 0.6) !important;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-left: 4px solid #3b82f6 !important; /* خط أزرق عيادي */
        border-radius: 10px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3) !important;
        border-left: 4px solid #10b981 !important; /* يتغير للأخضر عند التمرير */
    }

    /* الأزرار الرئيسية */
    .stButton>button {
        background-color: #3b82f6; /* أزرق احترافي */
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 20px;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #2563eb;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }

    /* الشارات (Badges) */
    .doc-id-badge {
        background-color: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        padding: 4px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.9em;
    }
    .score-badge {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.9em;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 🔌 دوال الاتصال بالخادم
# ==========================================
def execute_search(
    query: str, top_k: int, model: str, use_spell: bool, 
    use_synonyms: bool, use_clustering: bool, k1: float, b: float, alpha: float
) -> Optional[Dict[str, Any]]:
    payload = {
        "query": query, "top_k": top_k, "search_model": model,
        "use_spell_check": use_spell, "use_synonyms": use_synonyms, 
        "use_clustering": use_clustering, "k1": k1, "b": b, "alpha": alpha
    }
    try:
        response = requests.post(f"{API_BASE_URL}/search", json=payload, timeout=300)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"⚠️ فشل الاتصال بمحرك البحث: {e}")
        return None

def execute_rag(question: str, top_k: int, alpha: float) -> Optional[Dict[str, Any]]:
    try:
        response = requests.post(f"{API_BASE_URL}/ask", json={"question": question, "top_k": top_k, "alpha": alpha}, timeout=300)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"⚠️ فشل الاتصال بالمساعد الذكي: {e}")
        return None

def fetch_document(doc_id: str) -> Optional[Dict[str, Any]]:
    try:
        response = requests.get(f"{API_BASE_URL}/document/{doc_id}", timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"⚠️ تعذر جلب الوثيقة: {e}")
        return None

# ==========================================
# 🖼️ النافذة المنبثقة لقراءة الوثيقة (Popup Dialog)
# ==========================================
@st.dialog("📋 Clinical Record Viewer", width="large")
def show_document_viewer(doc_id: str) -> None:
    with st.spinner("Retrieving medical document..."):
        doc_data = fetch_document(doc_id.strip())
        if doc_data:
            st.markdown(f"<h3 style='color: #60a5fa;'>{doc_data.get('title', 'Untitled Medical Record')}</h3>", unsafe_allow_html=True)
            st.markdown(f"**Document ID:** <span class='doc-id-badge'>{doc_data.get('doc_id')}</span>", unsafe_allow_html=True)
            st.divider()
            st.text_area("Full Clinical Text:", value=doc_data.get("content", ""), height=400, disabled=True)
        else:
            st.error("❌ الوثيقة غير موجودة في قاعدة البيانات.")

# ==========================================
# 🖥️ مكونات الواجهة (UI Components)
# ==========================================
def build_sidebar() -> Dict[str, Any]:
    st.sidebar.markdown("<h2 style='text-align: center; color: #60a5fa;'>⚙️ Clinical Control</h2>", unsafe_allow_html=True)
    st.sidebar.divider()
    
    st.sidebar.markdown("<p style='color: #cbd5e1; font-weight: 600;'>🔍 Search Parameters</p>", unsafe_allow_html=True)
    model_selection = st.sidebar.selectbox("IR Engine Architecture", ["hybrid_parallel", "hybrid_serial", "bm25", "tfidf", "embeddings"])
    k_value = st.sidebar.slider("Top K Documents", 1, 50, 10)
    
    k1_val, b_val, alpha_val = 1.5, 0.75, 0.5
    
    if model_selection in ["bm25", "hybrid_parallel", "hybrid_serial"]:
        st.sidebar.markdown("<p style='color: #94a3b8; font-size: 13px;'>📊 BM25 Calibration</p>", unsafe_allow_html=True)
        k1_val = st.sidebar.slider("k1 (Term Saturation)", 0.0, 3.0, 1.5, 0.1)
        b_val = st.sidebar.slider("b (Length Normalization)", 0.0, 1.0, 0.75, 0.05)
        
    if model_selection == "hybrid_parallel":
        st.sidebar.markdown("<p style='color: #94a3b8; font-size: 13px;'>🧬 Hybrid Weights</p>", unsafe_allow_html=True)
        alpha_val = st.sidebar.slider("Alpha (BERT Weight)", 0.0, 1.0, 0.5, 0.1)
    
    st.sidebar.divider()
    spell_check = st.sidebar.toggle("🧬 Clinical Spell Correction", value=True)
    synonyms = st.sidebar.toggle("🔬 Semantic Synonyms", value=True)
    clustering = st.sidebar.toggle("🧫 Symptom Clustering", value=False)
    
    st.sidebar.divider()
    st.sidebar.markdown("<p style='color: #cbd5e1; font-weight: 600;'>🤖 RAG AI Settings</p>", unsafe_allow_html=True)
    rag_k_value = st.sidebar.slider("Context Documents", 1, 5, 3)
    rag_alpha = st.sidebar.slider("Hybrid Alpha (Weight)", 0.0, 1.0, 0.7)
    
    return {
        "model": model_selection, "top_k": k_value, "use_spell": spell_check,
        "use_synonyms": synonyms, "use_clustering": clustering,
        "k1": k1_val, "b": b_val, "alpha": alpha_val,
        "rag_top_k": rag_k_value, "rag_alpha": rag_alpha
    }

def main() -> None:
    init_session_state()
    inject_custom_css()
    config = build_sidebar()
    
    tab1, tab2 = st.tabs(["🔬 Epidemiological Search", "🩺 Medical AI Assistant"])
    
    with tab1:
        st.markdown('<p class="virus-title">CORD-19 Research Engine</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-title">Advanced Epidemiological & Viral Information Retrieval System</p>', unsafe_allow_html=True)
        
        user_query = st.text_input("Enter virus strains, symptoms, or medical keywords...", value=st.session_state.last_query)
        
        if st.button("🔍 INITIATE SCAN (بدء البحث المرجعي)", type="primary", use_container_width=True):
            if not user_query.strip():
                st.warning("⚠️ يرجى إدخال مصطلحات البحث أولاً.")
            else:
                with st.spinner("Searching medical literature..."):
                    results = execute_search(
                        query=user_query, top_k=config["top_k"], model=config["model"],
                        use_spell=config["use_spell"], use_synonyms=config["use_synonyms"], 
                        use_clustering=config["use_clustering"], k1=config["k1"], 
                        b=config["b"], alpha=config["alpha"]
                    )
                    # حفظ النتائج في ذاكرة الجلسة
                    if results:
                        st.session_state.search_results = results
                        st.session_state.last_query = user_query

        # عرض النتائج من ذاكرة الجلسة إن وجدت
        if st.session_state.search_results:
            results = st.session_state.search_results
            if "results" in results:
                st.markdown(f"<p style='color: #34d399; font-weight: 600; font-size: 18px;'>✅ Search Complete: {results.get('results_count', 0)} clinical records found.</p>", unsafe_allow_html=True)
                
                if results.get('refined_query') and results['refined_query'] != st.session_state.last_query:
                    st.info(f"💡 **Refined Medical Query:** {results['refined_query']}")
                
                for item in results["results"]:
                    with st.container(border=True):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**🦠 Record ID:** <span class='doc-id-badge'>{item['doc_id']}</span>", unsafe_allow_html=True)
                            st.markdown(f"**🎯 Relevance Score:** <span class='score-badge'>{item['score']:.4f}</span>", unsafe_allow_html=True)
                        with col2:
                            if st.button("📖 قراءة التقرير", key=f"btn_search_{item['doc_id']}", use_container_width=True):
                                show_document_viewer(item['doc_id'])

    with tab2:
        st.markdown('<p class="virus-title">Medical RAG Assistant</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-title">Clinical Q&A powered by AI & Hybrid Retrieval</p>', unsafe_allow_html=True)
        
        # عرض تاريخ الدردشة المحفوظ
        for msg_index, msg in enumerate(st.session_state.chat_history):
            with st.chat_message(msg["role"], avatar=msg["avatar"]):
                st.markdown(msg["content"], unsafe_allow_html=True)
                if "refs" in msg and msg["refs"]:
                    st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 10px 0;'>", unsafe_allow_html=True)
                    st.markdown("**📚 Source Documents (المراجع):**")
                    cols = st.columns(len(msg["refs"]))
                    for idx, ref in enumerate(msg["refs"]):
                        with cols[idx]:
                            if st.button(f"📄 {ref}", key=f"btn_hist_{msg_index}_{ref}", use_container_width=True):
                                show_document_viewer(ref)
        
        question = st.chat_input("Ask about COVID-19, virology, or clinical trials...")
        
        if question:
            # إضافة سؤال المستخدم إلى الذاكرة
            st.session_state.chat_history.append({"role": "user", "avatar": "🩺", "content": question})
            
            with st.chat_message("user", avatar="🩺"):
                st.write(question)
                
            with st.chat_message("assistant", avatar="🏥"):
                with st.spinner("Analyzing medical literature to generate consensus..."):
                    response = execute_rag(question, config["rag_top_k"], config["rag_alpha"])
                    
                    if response:
                        answer_text = response.get("answer", "No clinical data found to answer this query.")
                        refs = response.get("referenced_docs", [])
                        
                        st.markdown(answer_text)
                        
                        if refs:
                            st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 10px 0;'>", unsafe_allow_html=True)
                            st.markdown("**📚 Source Documents (المراجع):**")
                            cols = st.columns(len(refs))
                            for idx, ref in enumerate(refs):
                                with cols[idx]:
                                    if st.button(f"📄 {ref}", key=f"btn_new_{ref}", use_container_width=True):
                                        show_document_viewer(ref)
                        
                        # حفظ إجابة المساعد والمراجع في الذاكرة
                        st.session_state.chat_history.append({
                            "role": "assistant", 
                            "avatar": "🏥", 
                            "content": answer_text,
                            "refs": refs
                        })

if __name__ == "__main__":
    main()