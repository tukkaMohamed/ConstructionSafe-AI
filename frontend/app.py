import streamlit as st

from api_client import query_backend, check_health


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="ConstructionSafe AI",
    page_icon="🦺",
    layout="wide",
)


# --------------------------------------------------
# Custom styling
# --------------------------------------------------

st.markdown(
    """
    <style>

        /* Main title */
        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        /* Subtitle */
        .subtitle {
            color: #9ca3af;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }

        /* Source cards */
        .source-box {
            padding: 0.8rem 1rem;
            border-radius: 0.6rem;
            background-color: #262626;
            color: #ffffff !important;
            margin-bottom: 0.5rem;
            border: 1px solid #444444;
            font-size: 1rem;
        }

        /* YOLO detection cards */
        .detection-box {
            padding: 0.75rem 1rem;
            border-radius: 0.6rem;
            background-color: #262626;
            color: #ffffff !important;
            margin-bottom: 0.5rem;
            border: 1px solid #444444;
            font-size: 1rem;
        }

        .detection-box b {
            color: #ffffff !important;
        }

        /* Make text inside our custom boxes visible */
        .source-box,
        .source-box *,
        .detection-box,
        .detection-box * {
            color: #ffffff !important;
        }

    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🦺 ConstructionSafe AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Multimodal Construction Safety Assistant powered by RAG, YOLO and Ollama"
    "</div>",
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.header("System")

    if st.button("Check Backend Health"):
        try:
            health = check_health()

            if health.get("status") == "healthy":
                st.success("Backend is healthy")
            else:
                st.warning("Backend responded, but status is not healthy")

        except Exception as e:
            st.error(f"Backend connection failed: {e}")

    st.divider()

    st.markdown("### How to use")

    st.markdown(
        """
        1. Ask a construction safety question.
        2. Optionally upload a construction image.
        3. Click **Analyze**.
        4. Review the answer, sources and YOLO detections.
        """
    )


# --------------------------------------------------
# Input section
# --------------------------------------------------

st.subheader("Ask a Safety Question")

question = st.text_area(
    "Question",
    placeholder="Example: When should workers wear hard hats?",
    height=100,
)

uploaded_image = st.file_uploader(
    "Upload a construction image (optional)",
    type=["jpg", "jpeg", "png"],
)


# --------------------------------------------------
# Image preview
# --------------------------------------------------

if uploaded_image is not None:
    st.subheader("Uploaded Image")
    st.image(
        uploaded_image,
        caption=uploaded_image.name,
        use_container_width=True,
    )


# --------------------------------------------------
# Analyze
# --------------------------------------------------

if st.button(
    "🔍 Analyze",
    type="primary",
    use_container_width=True,
):

    if not question.strip():
        st.warning("Please enter a question first.")

    else:
        with st.spinner(
            "Analyzing the question and image..."
        ):
            try:
                result = query_backend(
                    question=question.strip(),
                    image=uploaded_image,
                )

                st.session_state["result"] = result

            except Exception as e:
                st.error(
                    f"Could not connect to the backend: {e}"
                )


# --------------------------------------------------
# Results
# --------------------------------------------------

if "result" in st.session_state:

    result = st.session_state["result"]

    st.divider()

    st.subheader("Answer")

    st.markdown(
        result.get(
            "answer",
            "No answer returned."
        )
    )

    # ----------------------------------------------
    # Sources
    # ----------------------------------------------

    sources = result.get("sources", [])

    if sources:
        st.subheader("📚 Sources")

        for source in sources:
            st.markdown(
                f'<div class="source-box">📄 {source}</div>',
                unsafe_allow_html=True,
            )

    # ----------------------------------------------
    # YOLO detections
    # ----------------------------------------------

    detections = result.get("detections", [])

    if detections:

        st.subheader("👁️ YOLO Detections")

        st.caption(
            "These are visual detections only and do not automatically "
            "prove safety compliance or non-compliance."
        )

        for detection in detections:

            class_name = detection.get(
                "class_name",
                "Unknown"
            )

            confidence = detection.get(
                "confidence",
                0
            )

            bbox = detection.get(
                "bbox",
                []
            )

            st.markdown(
                f"""
                <div class="detection-box">
                    <b>{class_name}</b>
                    &nbsp; | &nbsp;
                    Confidence: {confidence:.3f}
                    &nbsp; | &nbsp;
                    BBox: {bbox}
                </div>
                """,
                unsafe_allow_html=True,
            )

    elif uploaded_image is not None:

        st.info(
            "No YOLO detections were returned for this image."
        )