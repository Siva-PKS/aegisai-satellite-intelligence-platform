import streamlit as st
import cv2
import numpy as np
from PIL import Image
import faiss
from sentence_transformers import SentenceTransformer
import anthropic

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="AI Satellite Dashboard", layout="wide")

# -------------------------------
# UI Styling
# -------------------------------
st.markdown("""
<style>

/* Background */
body {
    background-color: #0e1117;
    color: white;
}

/* Metric cards */
.metric-box {
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    font-size: 18px;
    font-weight: bold;
    color: white;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.4);
}

/* Different colors */
.objects {
    background: linear-gradient(135deg, #2563eb, #1e3a8a);
}
.risk {
    background: linear-gradient(135deg, #dc2626, #7f1d1d);
}
.time {
    background: linear-gradient(135deg, #059669, #064e3b);
}

/* Buttons */
div.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 18px;
    font-weight: 600;
}
div.stButton > button:hover {
    background-color: #1d4ed8;
}

/* Sidebar button */
section[data-testid="stSidebar"] button {
    background-color: #2563eb !important;
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# Claude API (SAFE)
# -------------------------------
api_key = st.secrets.get("ANTHROPIC_API_KEY")

if not api_key:
    st.warning("⚠️ Claude API key not configured. Showing demo mode.")
    client = None
else:
    client = anthropic.Anthropic(api_key=api_key)

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
    if client is None:
        return "⚠️ Demo Mode: Claude API not configured."

    try:
        response = client.messages.create(
            model="claude-3-haiku-20240307",  # faster + safer
            max_tokens=300,
            messages=[
                {"role": "user",
                 "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
            ]
        )
        return response.content[0].text

    except Exception as e:
        return f"❌ Claude API Error: {str(e)}"

# -------------------------------
# Sidebar
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

# Initial placeholders
col1.markdown('<div class="metric-box objects">📍 Objects: --</div>', unsafe_allow_html=True)
col2.markdown('<div class="metric-box risk">⚠️ Risk Level: --</div>', unsafe_allow_html=True)
col3.markdown('<div class="metric-box time">⏱️ Response Time: --</div>', unsafe_allow_html=True)

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

        # Dynamic metrics update
        risk_level = "High" if len(objects) > 2 else "Medium"

        col1.markdown(f'<div class="metric-box objects">📍 Objects: {len(objects)}</div>', unsafe_allow_html=True)
        col2.markdown(f'<div class="metric-box risk">⚠️ Risk Level: {risk_level}</div>', unsafe_allow_html=True)
        col3.markdown('<div class="metric-box time">⏱️ Response Time: 2-5 sec</div>', unsafe_allow_html=True)

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
