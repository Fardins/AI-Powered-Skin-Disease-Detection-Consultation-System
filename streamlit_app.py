import streamlit as st
import tempfile
from predictor import Predictor
from llm_advice import generate_advice, llm

# Initialize predictor
predictor = Predictor()

st.set_page_config(page_title="Skin Disease AI", layout="wide")

st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>AI-powered skin disease detection and consultation system</h1>",
    unsafe_allow_html=True
)

# Session state
if "disease" not in st.session_state:
    st.session_state.disease = None
if "confidence" not in st.session_state:
    st.session_state.confidence = None
if "advice" not in st.session_state:
    st.session_state.advice = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "image" not in st.session_state:
    st.session_state.image = None

# Upload image
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    uploaded_file = st.file_uploader(
        "📤 Upload Image",
        type=["jpg", "jpeg", "png"]
    )

# Generate button
if st.button("🔍 Generate"):

    if uploaded_file is not None:
        # Save image in session (for UI)
        st.session_state.image = uploaded_file

        # Save temp file (for model)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name

        # Prediction
        disease, confidence = predictor.predict(temp_path)

        st.session_state.disease = disease
        st.session_state.confidence = confidence

        # LLM Advice
        advice = generate_advice(disease)
        st.session_state.advice = advice

        # ✅ Reset chat for new image
        st.session_state.messages = []

    else:
        st.warning("⚠️ Please upload an image first.")

# Show results in 2 columns
if st.session_state.disease:
    col1, col2 = st.columns([2, 1])

    # LEFT → Result + Advice
    with col1:
        st.subheader("🔍 Detection Result")
        st.write(f"**Disease:** {st.session_state.disease}")
        st.write(f"**Confidence:** {st.session_state.confidence:.2f}")

        st.subheader("💡 AI Medical Advice")
        st.write(st.session_state.advice)

    # RIGHT → Image
    with col2:
        st.subheader("🖼️ Uploaded Image")
        if st.session_state.image:
            st.image(st.session_state.image, width="stretch")

# ChatGPT-style Q&A
if st.session_state.disease:

    st.markdown("---")
    st.subheader("💬 Ask More Questions")

    # Show previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input (fixed bottom)
    user_input = st.chat_input("Ask anything about the disease...")

    if user_input:
        # User message
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        # LLM response
        prompt = f"""
You are a dermatologist AI.

Disease: {st.session_state.disease}

User Question: {user_input}

Answer clearly and helpfully.
"""
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = llm.invoke(prompt)
                answer = response.content
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

# Refresh button
if st.button("🔄 Refresh"):
    st.session_state.clear()
    st.rerun()