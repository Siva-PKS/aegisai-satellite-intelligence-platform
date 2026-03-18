import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
import faiss
from sentence_transformers import SentenceTransformer
import anthropic
import pandas as pd
import random
from ultralytics import YOLO

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="AegisAI Dashboard", layout="wide")

# -------------------------------
# UI Styling
# -------------------------------
st.markdown("""
<style>
body {background-color: #0e1117; color: white;}

.metric-box {
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    font-weight: bold;
    color: white;
}

.objects { background: linear-gradient(135deg, #2563eb, #1e3a8a); }
.risk { background: linear-gradient(135deg, #dc2626, #7f1d1d); }
.time { background: linear-gradient(135deg, #059669, #064e3b); }

div.stButton > button {
    background-color: #2563eb;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Claude Setup (Hybrid)
# -------------------------------
api_key = st.secrets.get("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=api_key) if api_key else None

# -------------------------------
# Load YOLO Model (Cached)
# -------------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

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
# YOLO Detection
# -------------------------------
def detect_objects(image):
    results = model(image)

    objects = []
    boxes = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            objects.append(label)
            boxes.append((x1, y1, x2, y2))

    return objects, boxes

# -------------------------------
# Draw Boxes
# -------------------------------
def draw_boxes(image, boxes, objects):
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)

    for box, label in zip(boxes, objects):
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[1] - 10), label, fill="red")

    return img

# -------------------------------
# RAG Retrieval
# -------------------------------
def retrieve_context(query):
    q_embed = embed_model.encode([query])
    D, I = index.search(np.array(q_embed), k=2)
    return "\n".join([documents[i] for i in I[0]])

# -------------------------------
# Multi-Agent System
# -------------------------------
def planner_agent(objects):
    return f"Plan: Analyze {len(objects)} detected objects."

def analyst_agent(context):
    return f"Analysis: {context}"

def decision_agent(context):
    if "convoy" in context.lower():
        return "High Risk: Coordinated movement detected."
    return "Medium Risk: Monitor situation."

# -------------------------------
# Hybrid AI (Claude + Fallback)
# -------------------------------
def ask_claude(context, query):
    if client is None:
        return decision_agent(context)

    try:
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=200,
            messages=[
                {"role": "user",
                 "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
            ]
        )
        return response.content[0].text

    except:
        return decision_agent(context)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("🛰️ Control Panel")

uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg","png"])
query = st.sidebar.text_input("Ask Question", "Is this a threat?")
run = st.sidebar.button("🚀 Run Analysis")

# -------------------------------
# Dashboard
# -------------------------------
st.title("🛰️ AegisAI Intelligence Dashboard")

col1, col2, col3 = st.columns(3)

col1.markdown('<div class="metric-box objects">📍 Objects: --</div>', unsafe_allow_html=True)
col2.markdown('<div class="metric-box risk">⚠️ Risk Level: --</div>', unsafe_allow_html=True)
col3.markdown('<div class="metric-box time">⏱️ Response Time: --</div>', unsafe_allow_html=True)

# -------------------------------
# PROCESS
# -------------------------------
if uploaded_file:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    st.subheader("🛰️ Detection View")

    if run:
        objects, boxes = detect_objects(img_array)
        boxed_img = draw_boxes(img_array, boxes, objects)

        context = retrieve_context(" ".join(objects))
        response = ask_claude(context, query)

        plan = planner_agent(objects)
        analysis = analyst_agent(context)
        decision = decision_agent(context)

        # Metrics
        risk_level = "High" if len(objects) > 3 else "Medium"

        col1.markdown(f'<div class="metric-box objects">📍 Objects: {len(objects)}</div>', unsafe_allow_html=True)
        col2.markdown(f'<div class="metric-box risk">⚠️ Risk: {risk_level}</div>', unsafe_allow_html=True)
        col3.markdown('<div class="metric-box time">⏱️ 2-5 sec</div>', unsafe_allow_html=True)

        st.image(boxed_img, use_column_width=True)

        left, right = st.columns(2)

        with left:
            st.subheader("🧠 Multi-Agent Reasoning")
            st.write(plan)
            st.write(analysis)
            st.write(decision)

        with right:
            st.subheader("🤖 AI Decision")
            st.success(response)

        # Risk Trend Chart
        st.subheader("📊 Risk Trend")
        data = pd.DataFrame({
            "Time": ["T1","T2","T3","T4"],
            "Risk Score": [random.randint(40,90) for _ in range(4)]
        })
        st.line_chart(data.set_index("Time"))

        # Map
        st.subheader("🗺️ Location Tracking")
        map_data = pd.DataFrame({
            'lat': [20 + random.random()],
            'lon': [78 + random.random()]
        })
        st.map(map_data)
