import streamlit as st
from pathlib import Path
import time

from src.vector_store import get_collection
from src.rag_pipeline import answer_query


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Multimodal RAG AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(99,102,241,0.15), transparent 30%),
        radial-gradient(circle at 90% 20%, rgba(168,85,247,0.12), transparent 30%),
        radial-gradient(circle at 50% 90%, rgba(14,165,233,0.10), transparent 30%),
        #070b14;
    color: #f8fafc;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
    max-width: 1400px;
}


/* =========================================================
   HERO
   ========================================================= */

.hero {
    padding: 45px 30px;
    border-radius: 28px;
    background:
        linear-gradient(
            135deg,
            rgba(30,41,59,0.90),
            rgba(15,23,42,0.78)
        );
    border: 1px solid rgba(148,163,184,0.15);
    box-shadow:
        0 25px 70px rgba(0,0,0,0.35),
        inset 0 1px 0 rgba(255,255,255,0.05);
    margin-bottom: 30px;
}

.hero-badge {
    display: inline-block;
    padding: 7px 14px;
    border-radius: 999px;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(129,140,248,0.25);
    color: #a5b4fc;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 18px;
}

.hero-title {
    font-size: 46px;
    font-weight: 800;
    letter-spacing: -1.5px;
    margin: 0;
    background: linear-gradient(
        90deg,
        #ffffff,
        #c4b5fd,
        #93c5fd
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 17px;
    line-height: 1.7;
    max-width: 900px;
    margin-top: 15px;
}


/* =========================================================
   FEATURE CARDS
   ========================================================= */

.feature-card {
    padding: 22px;
    border-radius: 20px;
    background: rgba(15,23,42,0.70);
    border: 1px solid rgba(148,163,184,0.12);
    min-height: 150px;
    transition: all 0.3s ease;
}

.feature-card:hover {
    transform: translateY(-4px);
    border-color: rgba(129,140,248,0.4);
    box-shadow: 0 15px 35px rgba(0,0,0,0.25);
}

.feature-icon {
    font-size: 28px;
    margin-bottom: 10px;
}

.feature-title {
    font-size: 16px;
    font-weight: 700;
    color: #f8fafc;
}

.feature-text {
    font-size: 13px;
    color: #94a3b8;
    margin-top: 6px;
    line-height: 1.6;
}


/* =========================================================
   SECTION TITLES
   ========================================================= */

.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #f8fafc;
    margin-top: 30px;
    margin-bottom: 15px;
}


/* =========================================================
   ANSWER CARD
   ========================================================= */

.answer-card {
    padding: 28px;
    border-radius: 22px;
    background: rgba(15,23,42,0.80);
    border: 1px solid rgba(129,140,248,0.20);
    box-shadow: 0 20px 50px rgba(0,0,0,0.25);
    line-height: 1.8;
}


/* =========================================================
   STATUS
   ========================================================= */

.status-online {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #86efac;
    font-size: 13px;
    font-weight: 600;
}

.status-dot {
    width: 8px;
    height: 8px;
    background: #22c55e;
    border-radius: 50%;
    box-shadow: 0 0 10px #22c55e;
}


/* =========================================================
   SIDEBAR
   ========================================================= */

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            rgba(15,23,42,0.98),
            rgba(7,11,20,0.98)
        );
    border-right: 1px solid rgba(148,163,184,0.10);
}

.sidebar-title {
    font-size: 20px;
    font-weight: 800;
    color: #f8fafc;
}

.sidebar-text {
    color: #94a3b8;
    font-size: 13px;
    line-height: 1.6;
}


/* =========================================================
   TEXT INPUT
   ========================================================= */

.stTextArea textarea {
    background: rgba(15,23,42,0.85) !important;
    color: #f8fafc !important;
    border: 1px solid rgba(129,140,248,0.25) !important;
    border-radius: 16px !important;
    font-size: 15px !important;
}

.stTextArea textarea:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 1px #818cf8 !important;
}


/* =========================================================
   BUTTON
   ========================================================= */

.stButton > button {
    width: 100%;
    border-radius: 14px;
    border: 1px solid rgba(129,140,248,0.30);
    background: linear-gradient(
        135deg,
        #6366f1,
        #8b5cf6
    );
    color: white;
    font-weight: 700;
    padding: 12px 20px;
    transition: all 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 30px rgba(99,102,241,0.35);
    border-color: #a5b4fc;
}


/* =========================================================
   METRICS
   ========================================================= */

[data-testid="stMetric"] {
    background: rgba(15,23,42,0.65);
    padding: 16px;
    border-radius: 16px;
    border: 1px solid rgba(148,163,184,0.10);
}


/* =========================================================
   IMAGE
   ========================================================= */

.stImage img {
    border-radius: 18px;
    border: 1px solid rgba(148,163,184,0.15);
}


/* =========================================================
   FOOTER
   ========================================================= */

.footer {
    text-align: center;
    color: #64748b;
    font-size: 12px;
    margin-top: 50px;
    padding-top: 20px;
    border-top: 1px solid rgba(148,163,184,0.08);
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# LOAD CHROMADB
# ============================================================

@st.cache_resource
def load_collection():
    return get_collection()


collection = load_collection()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
<div class="sidebar-title">
🧠 Multimodal RAG
</div>
""",
        unsafe_allow_html=True
    )

    st.markdown(
        """
<div class="sidebar-text">
An intelligent retrieval system that understands
both <b>documents</b> and <b>images</b>.
</div>
""",
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### ⚙️ System Status")

    st.markdown(
        """
<div class="status-online">
<span class="status-dot"></span>
Gemini AI Online
</div>
<div class="status-online">
        <span class="status-dot"></span>
        Flask API Online
    </div>
""",
        unsafe_allow_html=True
    )

    st.write("")

    st.markdown(
        """
<div class="status-online">
<span class="status-dot"></span>
ChromaDB Connected
</div>
""",
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### 📚 Knowledge Base")

    total_records = collection.count()

    st.metric(
        label="Indexed Sources",
        value=total_records
    )

    st.caption(
        "PDF chunks and images stored in the vector database."
    )

    st.markdown("---")

    st.markdown("### 🧩 Architecture")

    st.markdown(
        """
**User Query**

↓

🔍 **Semantic Retrieval**

↓

🗄️ **ChromaDB**

↓

📄 **PDF + 🖼️ Image**

↓

✨ **Gemini Multimodal AI**

↓

💬 **Answer**
"""
    )

    st.markdown("---")

    st.markdown("### 🔧 Retrieval Settings")

    sidebar_pdf_top_k = st.slider(
        "📄 PDF Results",
        min_value=1,
        max_value=8,
        value=5
    )

    sidebar_image_top_k = st.slider(
        "🖼️ Image Results",
        min_value=1,
        max_value=3,
        value=1
    )

    st.markdown("---")

    st.caption(
        "Built with LangChain • Gemini • ChromaDB • Streamlit"
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
<div class="hero">

<div class="hero-badge">
✦ AI-POWERED MULTIMODAL KNOWLEDGE SYSTEM
</div>

<h1 class="hero-title">
Multimodal RAG AI
</h1>

<div class="hero-subtitle">
Ask questions about your documents and images.
The system retrieves the most relevant information
from the knowledge base and lets Gemini understand
both text and visual context.
</div>

</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# FEATURE CARDS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        """
<div class="feature-card">

<div class="feature-icon">
📄
</div>

<div class="feature-title">
Document Intelligence
</div>

<div class="feature-text">
Search and understand information stored inside PDF
documents.
</div>

</div>
""",
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        """
<div class="feature-card">

<div class="feature-icon">
🖼️
</div>

<div class="feature-title">
Visual Intelligence
</div>

<div class="feature-text">
Retrieve relevant images and allow Gemini to analyze
their visual content.
</div>

</div>
""",
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        """
<div class="feature-card">

<div class="feature-icon">
✨
</div>

<div class="feature-title">
Grounded Answers
</div>

<div class="feature-text">
Generate responses using retrieved knowledge instead
of guessing.
</div>

</div>
""",
        unsafe_allow_html=True
    )


# ============================================================
# QUERY SECTION
# ============================================================

st.markdown(
    '<div class="section-title">💬 Ask Your Knowledge Base</div>',
    unsafe_allow_html=True
)

query = st.text_area(
    label="Your question",
    placeholder=(
        "Ask something about the PDF or image...\n\n"
        "Example: What technologies are discussed in the paper?\n"
        "Example: What is shown in the image?"
    ),
    height=130,
    label_visibility="collapsed"
)


# ============================================================
# QUERY BUTTON
# ============================================================

ask_button = st.button(
    "✨ Generate Intelligent Answer",
    use_container_width=True
)


# ============================================================
# PROCESS QUERY
# ============================================================

if ask_button:

    if not query.strip():

        st.warning(
            "⚠️ Please enter a question before generating an answer."
        )

    else:

        with st.spinner(
            "🔍 Searching the knowledge base and consulting Gemini..."
        ):

            try:

                start_time = time.time()

                result = answer_query(
                    collection=collection,
                    query=query,
                    pdf_top_k=sidebar_pdf_top_k,
                    image_top_k=sidebar_image_top_k
                )

                elapsed_time = time.time() - start_time

                st.session_state["result"] = result
                st.session_state["elapsed_time"] = elapsed_time
                st.session_state["query"] = query

            except Exception as e:

                st.error(
                    "❌ Something went wrong while processing your question."
                )

                st.exception(e)


# ============================================================
# DISPLAY RESULTS
# ============================================================

if "result" in st.session_state:

    result = st.session_state["result"]

    elapsed_time = st.session_state.get(
        "elapsed_time",
        0
    )

    # --------------------------------------------------------
    # Safely get result data
    # --------------------------------------------------------

    pdf_results = result.get(
        "pdf_results",
        []
    )

    image_results = result.get(
        "image_results",
        []
    )

    image_sources = result.get(
        "image_sources",
        []
    )

    answer = result.get(
        "answer",
        "No answer was generated."
    )


    # ========================================================
    # RESULT DIVIDER
    # ========================================================

    st.markdown("---")


    # ========================================================
    # AI RESPONSE
    # ========================================================

    st.markdown(
        '<div class="section-title">🧠 AI Response</div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # METRICS
    # ========================================================

    metric1, metric2, metric3 = st.columns(3)

    with metric1:

        st.metric(
            "📄 PDF Sources",
            len(pdf_results)
        )

    with metric2:

        st.metric(
            "🖼️ Image Sources",
            len(image_results)
        )

    with metric3:

        st.metric(
            "⚡ Response Time",
            f"{elapsed_time:.2f}s"
        )


    st.write("")


    # ========================================================
    # ANSWER
    # ========================================================

    st.markdown(
        '<div class="answer-card">',
        unsafe_allow_html=True
    )

    st.markdown(answer)

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


    # ========================================================
    # RETRIEVED IMAGES
    # ========================================================

    if image_sources:

        st.markdown(
            '<div class="section-title">🖼️ Retrieved Visual Context</div>',
            unsafe_allow_html=True
        )

        image_columns = st.columns(
            len(image_sources)
        )

        for index, image_path in enumerate(image_sources):

            with image_columns[index]:

                path = Path(image_path)

                if path.exists():

                    st.image(
                        str(path),
                        caption=f"Retrieved Image {index + 1}",
                        use_container_width=True
                    )

                    st.caption(
                        f"Source: {path.name}"
                    )

                else:

                    st.warning(
                        f"Image not found: {image_path}"
                    )


    # ========================================================
    # RETRIEVED PDF CONTEXT
    # ========================================================

    if pdf_results:

        st.markdown(
            '<div class="section-title">📚 Retrieved PDF Context</div>',
            unsafe_allow_html=True
        )

        st.caption(
            f"{len(pdf_results)} relevant PDF sections retrieved "
            "from the knowledge base."
        )


        for index, pdf_result in enumerate(pdf_results):

            metadata = pdf_result.get(
                "metadata",
                {}
            )

            page = metadata.get(
                "page_label",
                "Unknown"
            )

            content = pdf_result.get(
                "content",
                ""
            )

            distance = pdf_result.get(
                "distance",
                None
            )


            # ------------------------------------------------
            # Native Streamlit container
            # ------------------------------------------------

            with st.container(border=True):

                source_col, page_col = st.columns(
                    [4, 1]
                )

                with source_col:

                    st.markdown(
                        f"### 📄 SOURCE {index + 1}"
                    )

                with page_col:

                    st.markdown(
                        f"**Page {page}**"
                    )


                st.markdown("---")


                # PDF content
                st.write(content)


                # Retrieval distance
                if distance is not None:

                    st.caption(
                        f"Retrieval distance: {distance:.4f}"
                    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="footer">

🧠 <b>Multimodal RAG AI</b>

<br><br>

Retrieval-Augmented Generation powered by
Gemini + ChromaDB + LangChain

<br><br>

<span style="opacity:0.7;">
Understanding text. Understanding images. Understanding context.
</span>

</div>
""",
    unsafe_allow_html=True
)