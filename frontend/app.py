import textwrap

import streamlit as st
from api_client import query_backend, check_health


# =========================================================
# HELPERS
# =========================================================

def render_html(html: str) -> None:
    st.html(textwrap.dedent(html))


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="ConstructionSafe AI",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM CSS — dark theme + brand accent
# =========================================================

render_html(
    """
    <style>

    /* ================= GLOBAL ================= */

    .stApp {
        background: #0b0f14;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .main .block-container {
        max-width: 900px;
        padding-top: 1.2rem;
        padding-bottom: 6rem;
    }


    /* ================= SIDEBAR ================= */

    [data-testid="stSidebar"] {
        background: #11161d;
        border-right: 1px solid #252c35;
    }

    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #f5f5f5;
    }

    .sidebar-subtitle {
        color: #9ca3af;
        font-size: 0.82rem;
        margin-top: 0.2rem;
        line-height: 1.5;
    }


    /* ================= HEADER ================= */

    .brand-row {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 0.3rem;
    }

    .brand-title {
        font-size: 1.7rem;
        font-weight: 750;
        line-height: 1.15;
        margin: 0;
        color: #f5f5f5;
    }

    .brand-title span {
        color: #ff5a5f;
    }

    .brand-subtitle {
        color: #9ca3af;
        font-size: 0.88rem;
        margin: 0.15rem 0 1.2rem 0;
    }


    /* ================= CHAT MESSAGES ================= */

    [data-testid="stChatMessage"] {
        background: #171d24;
        border: 1px solid #232a33;
        border-radius: 0.9rem;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.7rem;
    }

    [data-testid="stChatMessageAvatarUser"] {
        background: #ff5257 !important;
    }

    [data-testid="stChatMessageAvatarAssistant"] {
        background: #20252d !important;
    }


    /* ================= CHAT INPUT (bottom bar) ================= */

    [data-testid="stChatInput"] {
        background: #11161d;
        border-top: 1px solid #252c35;
    }

    [data-testid="stChatInput"] textarea {
        background-color: #20252d !important;
        color: #f5f5f5 !important;
        border-radius: 0.85rem !important;
    }


    /* ================= MISC WIDGETS ================= */

    [data-testid="stFileUploader"] {
        background: #20252d;
        border: 1px dashed #555e69;
        border-radius: 0.7rem;
    }

    div.stButton > button {
        border-radius: 0.6rem;
        font-weight: 600;
    }

    /* Sources / detections chips inside an assistant message */
    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-top: 0.5rem;
    }

    .chip {
        background: #20252d;
        border: 1px solid #2a323d;
        border-radius: 999px;
        padding: 0.25rem 0.7rem;
        font-size: 0.78rem;
        color: #d1d5db;
    }

    .chip.detection {
        border-color: #3a4552;
    }

    .chip .conf {
        color: #9ca3af;
        margin-left: 0.35rem;
    }

    .pending-image-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: #20252d;
        border: 1px solid #2a323d;
        border-radius: 999px;
        padding: 0.3rem 0.8rem 0.3rem 0.5rem;
        font-size: 0.82rem;
        color: #d1d5db;
    }

    </style>
    """
)


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_image" not in st.session_state:
    st.session_state.pending_image = None


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    render_html(
        """
        <div class="sidebar-title">🦺 ConstructionSafe AI</div>
        <div class="sidebar-subtitle">
            Multimodal construction safety assistant
        </div>
        """
    )

    st.write("")

    if st.button("＋ New chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_image = None
        st.rerun()

    st.divider()

    st.markdown("**Combines**")
    st.markdown("📚 RAG &nbsp;·&nbsp; 👁️ YOLO11n &nbsp;·&nbsp; 🤖 Llama 3.2 &nbsp;·&nbsp; ⚡ FastAPI")

    st.divider()

    st.markdown("**Backend**")

    if st.button("Check backend", use_container_width=True):
        try:
            check_health()
            st.success("Backend connected", icon="✅")
        except Exception:
            st.error("Backend unavailable", icon="⚠️")


# =========================================================
# HEADER
# =========================================================

render_html(
    """
    <div class="brand-row">
        <div style="font-size:2.4rem;">🦺</div>
        <div>
            <div class="brand-title">ConstructionSafe <span>AI</span></div>
        </div>
    </div>
    """
)

st.markdown(
    '<div class="brand-subtitle">Ask a construction safety question, '
    'optionally with a site photo attached.</div>',
    unsafe_allow_html=True,
)


# =========================================================
# CHAT HISTORY
# =========================================================

if not st.session_state.messages:
    st.info("Start a conversation — ask a safety question or attach an image below.")

for message in st.session_state.messages:

    role = message["role"]
    avatar = "🧑‍🏭" if role == "user" else "🦺"

    with st.chat_message(role, avatar=avatar):

        if role == "user":
            st.markdown(message["content"])
            if message.get("image") is not None:
                st.image(message["image"], width=320)

        else:
            st.markdown(message["answer"])

            sources = message.get("sources", [])
            if sources:
                chips = "".join(
                    f'<span class="chip">📄 {s}</span>' for s in sources
                )
                render_html(f'<div class="chip-row">{chips}</div>')

            detections = message.get("detections", [])
            if detections:
                chips = "".join(
                    f'<span class="chip detection">🔎 {d.get("class_name", "Unknown")}'
                    f'<span class="conf">{d.get("confidence", 0):.2f}</span></span>'
                    for d in detections
                )
                render_html(f'<div class="chip-row">{chips}</div>')


# =========================================================
# PENDING IMAGE PREVIEW (shown above the input bar)
# =========================================================

if st.session_state.pending_image is not None:

    preview_col, clear_col = st.columns([6, 1])

    with preview_col:
        st.image(st.session_state.pending_image, width=140)

    with clear_col:
        if st.button("✕ Remove", key="remove_pending_image"):
            st.session_state.pending_image = None
            st.rerun()


# =========================================================
# ATTACH IMAGE (popover, keeps the input bar clean)
# =========================================================

attach_col, _ = st.columns([1, 5])

with attach_col:
    with st.popover("📎 Attach image"):
        uploaded = st.file_uploader(
            "Upload a construction site photo",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key="image_uploader",
        )
        if uploaded is not None:
            st.session_state.pending_image = uploaded


# =========================================================
# CHAT INPUT (bottom bar — press Enter or click send)
# =========================================================

question = st.chat_input("Ask anything about construction safety...")


# =========================================================
# PROCESS REQUEST
# =========================================================

if question:

    image_bytes = None
    if st.session_state.pending_image is not None:
        image_bytes = st.session_state.pending_image.getvalue()

    # ---------------- USER MESSAGE ----------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
            "image": image_bytes,
        }
    )

    # ---------------- API REQUEST ----------------

    with st.spinner("ConstructionSafe AI is analyzing..."):

        try:
            result = query_backend(
                question,
                image=st.session_state.pending_image,
            )

            answer = result.get("answer", "No answer was returned.")
            sources = result.get("sources", [])
            detections = result.get("detections", [])

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "answer": answer,
                    "sources": sources,
                    "detections": detections,
                }
            )

        except Exception as error:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "answer": f"⚠️ Something went wrong: {error}",
                    "sources": [],
                    "detections": [],
                }
            )

    st.session_state.pending_image = None
    st.rerun()