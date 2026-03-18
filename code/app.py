import streamlit as st
import cv2
import numpy as np
from PIL import Image
import faiss
from sentence_transformers import SentenceTransformer
import anthropic

# -------------------------------
# Page Config (Dark Dashboard)
# -------------------------------
st.set_page_config(page_title="AI Satellite Dashboard", layout="wide")

# Custom CSS (Palantir style)
st.markdown("""
<style>

/* Background */
body {
    background-color: #0e1117;
    color: white;
}

/* Metric cards */
.metric-box {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}

/* 🔥 Buttons (Premium Blue) */
div.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 18px;
    font-weight: 600;
    transition: 0.3s;
}

/* Hover effect */
div.stButton > button:hover {
    background-color: #1d4ed8;
    color: white;
}

/* Sidebar button fix */
section[data-testid="stSidebar"] button {
    background-color: #2563eb !important;
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# Claude API
# -------------------------------
client = anthropic.Anthropic(api_key="YOUR_API_KEY")

# -------------------------------
# RAG Setup
# -------------------------------
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

documents = [
    "Vehicle detected in restricted zone is high risk",
    "Clustered movement indicates convoy",
    "Isolated structure may be bunker or building",
    "Multiple vehicles indicate coordinated activity"
]

doc_embeddings = embed_model.encode(documents)
index = faiss.IndexFlatL2(384)
index.add(np.array(doc_embeddings))

# -------------------------------
# Functions
# -------------------------------
def detect_objects(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    objects = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            objects.append("Vehicle/Object")

    return objects

def retrieve_context(query):
    q_embed = embed_model.encode([query])
    D, I = index.search(np.array(q_embed), k=2)
    return "\n".join([documents[i] for i in I[0]])

def ask_claude(context, query):
    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=300,
        messages=[
            {"role": "user",
             "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
        ]
    )
    return response.content[0].text

# -------------------------------
# Sidebar (Control Panel)
# -------------------------------
st.sidebar.title("🛰️ Control Panel")

uploaded_file = st.sidebar.file_uploader("Upload Satellite Image", type=["jpg", "png"])
query = st.sidebar.text_input("Ask Intelligence Question", "Is this a threat?")
run = st.sidebar.button("🚀 Run Analysis")

# -------------------------------
# Main Dashboard
# -------------------------------
st.title("🛰️ AI Satellite Intelligence Dashboard")

col1, col2, col3 = st.columns(3)

# Metrics placeholders
with col1:
    st.markdown('<div class="metric-box">📍 Objects: --</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-box">⚠️ Risk Level: --</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-box">⏱️ Response Time: --</div>', unsafe_allow_html=True)

# -------------------------------
# Processing
# -------------------------------
if uploaded_file:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    st.subheader("🛰️ Satellite Image")
    st.image(image, use_column_width=True)

    if run:
        with st.spinner("Analyzing data..."):
            objects = detect_objects(img_array)
            context = retrieve_context(" ".join(objects))
            response = ask_claude(context, query)

        # Metrics Update
        col1.metric("📍 Objects", len(objects))
        col2.metric("⚠️ Risk Level", "High" if len(objects) > 2 else "Medium")
        col3.metric("⏱️ Response", "2-5 sec")

        # Layout sections
        left, right = st.columns(2)

        with left:
            st.subheader("🧠 Detected Objects")
            st.write(objects)

            st.subheader("📚 Intelligence Context (RAG)")
            st.write(context)

        with right:
            st.subheader("🤖 AI Decision (Claude)")
            st.success(response)
