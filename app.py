import streamlit as st
import pandas as pd
import requests
from typing import List, Dict, TypedDict, Any

# ================= BACKEND CONFIG =================
BACKEND_BASE = "http://127.0.0.1:8000"
PREDICT_URL = f"{BACKEND_BASE}/api/v1/predict"

# ================= STREAMLIT CONFIG =================
st.set_page_config(
    page_title="GenAI News Intelligence System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
.big-title {font-size:28px; font-weight:bold;}
.card {
    padding:15px;
    border-radius:12px;
    background: linear-gradient(135deg, #1f2937, #111827);
    color: white;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.green {color:#22c55e;}
.red {color:#ef4444;}
.orange {color:#f59e0b;}
</style>
""", unsafe_allow_html=True)

st.title("📰 GenAI News Intelligence System")
st.markdown("---")

# ================= TYPES =================
class HistoryItem(TypedDict):
    preview: str
    prediction: str
    confidence: float

# ================= SESSION =================
if "analysis_history" not in st.session_state:
    st.session_state["analysis_history"] = []

if "news_text" not in st.session_state:
    st.session_state["news_text"] = ""

def clear_text():
    st.session_state["news_text"] = ""

# ================= COLOR HELPER =================
def get_color(label):
    if label == "REAL":
        return "green"
    elif label == "FAKE":
        return "red"
    elif label == "SUSPICIOUS":
        return "orange"
    return ""

# ================= LAYOUT =================
col1, col2 = st.columns([2, 1])

# ================= LEFT PANEL =================
with col1:
    st.subheader("📝 Enter News Article")

    news_text = st.text_area(
        "Paste news content here",
        height=260,
        placeholder="Paste a complete news article...",
        key="news_text"
    )

    colb1, colb2 = st.columns(2)

    analyze = colb1.button(
        "🔍 Analyze Article",
        type="primary",
        use_container_width=True,
        disabled=len(news_text.strip()) < 50
    )

    colb2.button("🗑️ Clear", use_container_width=True, on_click=clear_text)

    if analyze:
        with st.spinner("🔄 AI analyzing (ML + RAG + OpenAI)..."):

            try:
                response = requests.post(
                    PREDICT_URL,
                    json={"text": news_text},
                    timeout=30
                )

                if response.status_code != 200:
                    st.error("❌ Backend error")
                    st.stop()

                data: Dict[str, Any] = response.json()

            except:
                st.error("❌ Backend not reachable")
                st.stop()

        # ================= RESULTS =================
        st.markdown("### 📊 Analysis Results")

        final = data.get("final_prediction", "UNKNOWN")
        color_class = get_color(final)

        st.markdown(
            f"<div class='card'><h2 class='{color_class}'>{final}</h2></div>",
            unsafe_allow_html=True
        )

        st.markdown(f"**Status:** {data.get('verification_status')}")

        # ================= METRICS =================
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Model Prediction", data["prediction"])
        r2.metric("Confidence", f'{data["confidence"]}%')
        r3.metric("Category", data["category"])
        r4.metric("Risk", data["warning"])

        # ================= EXPLANATION =================
        st.markdown("### 🧠 Key Indicators")
        for reason in data["explanation"]:
            st.write(f"• {reason}")

        # ================= AI EXPLANATION =================
        st.markdown("### 🤖 AI Explanation")
        st.info(data.get("ai_explanation", "Not available"))

        # ================= RAG =================
        st.markdown("### 🌐 Fact Check (RAG)")
        rag = data.get("rag_verification", {})
        st.write(f"**Verdict:** {rag.get('verdict')}")
        st.write(f"**Confidence:** {rag.get('confidence')}%")
        st.write(rag.get("explanation"))

        # ================= REALTIME =================
        st.markdown("### 🔎 Sources Found")
        realtime = data.get("realtime_verification", {})

        for r in realtime.get("results", [])[:5]:
            if r.get("trusted"):
                st.success(f"✅ {r['title']}")
            else:
                st.warning(f"⚠️ {r['title']}")

        # ================= HISTORY =================
        st.session_state["analysis_history"].insert(0, {
            "preview": news_text[:50] + "...",
            "prediction": final,
            "confidence": float(data["confidence"])
        })

# ================= RIGHT PANEL =================
with col2:
    st.subheader("📤 Upload CSV")

    uploaded = st.file_uploader("Upload CSV with text column", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df)} rows")

        text_col = next((c for c in df.columns if c in ["text","content","article","news","title"]), None)

        if not text_col:
            st.error("No text column found")
        else:
            sample = st.slider("Articles to analyze", 1, min(20, len(df)), 5)

            if st.button("📊 Analyze Batch", use_container_width=True):
                results = []
                bar = st.progress(0)

                for i, txt in enumerate(df[text_col].head(sample)):
                    if isinstance(txt, str) and len(txt.strip()) > 30:
                        r = requests.post(PREDICT_URL, json={"text": txt}).json()

                        results.append({
                            "Article": i + 1,
                            "Final": r.get("final_prediction"),
                            "Confidence": f"{r.get('confidence')}%",
                            "Category": r.get("category")
                        })

                    bar.progress((i + 1) / sample)

                st.dataframe(pd.DataFrame(results), use_container_width=True)

# ================= DASHBOARD =================
st.markdown("---")
st.subheader("📊 System Dashboard")

try:
    requests.get(BACKEND_BASE, timeout=3)
    backend_status = "✅ Active"
except:
    backend_status = "❌ Down"

d1, d2, d3 = st.columns(3)
d1.metric("Backend Status", backend_status)
d2.metric("Model", "ML + RAG Hybrid")
d3.metric("AI", "OpenAI")

# ================= HISTORY =================
if st.session_state["analysis_history"]:
    st.markdown("### 🕒 Recent Analyses")
    for h in st.session_state["analysis_history"][:3]:
        with st.expander(h["preview"]):
            st.write(f"Prediction: {h['prediction']}")
            st.write(f"Confidence: {h['confidence']}%")

# ================= SIDEBAR =================
st.sidebar.title("ℹ️ About")
st.sidebar.info("""
GenAI News Intelligence System

🔥 Features:
- Fake News Detection
- RAG Fact Checking
- AI Explanation
- Hybrid Decision

Tech:
FastAPI + Streamlit + OpenAI
""")

st.sidebar.markdown("---")
st.sidebar.write("Version: 2.0.0")