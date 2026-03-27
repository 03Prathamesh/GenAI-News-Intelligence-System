import streamlit as st
import pandas as pd
import requests
from typing import List, Dict, TypedDict, Any

# ================= BACKEND CONFIG =================
BACKEND_BASE = "http://127.0.0.1:8000"
PREDICT_URL = f"{BACKEND_BASE}/predict"

# ================= STREAMLIT CONFIG =================
st.set_page_config(
    page_title="GenAI News Intelligence System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📰 GenAI News Intelligence System")
st.markdown("---")

# ================= TYPES =================
class HistoryItem(TypedDict):
    preview: str
    prediction: str
    confidence: float

# ================= SESSION STATE =================
if "analysis_history" not in st.session_state:
    st.session_state["analysis_history"] = []

if "news_text" not in st.session_state:
    st.session_state["news_text"] = ""

def clear_text() -> None:
    st.session_state["news_text"] = ""

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

    colb2.button(
        "🗑️ Clear",
        use_container_width=True,
        on_click=clear_text
    )

    if analyze:
        progress = st.progress(0)
        status = st.empty()

        status.write("🔄 Sending text to backend...")
        progress.progress(20)

        try:
            response = requests.post(
                PREDICT_URL,
                json={"text": news_text},
                timeout=30
            )

            progress.progress(50)
            status.write("🧠 Running ML model...")

            if response.status_code != 200:
                st.error("❌ Backend error")
                st.stop()

            data: Dict[str, Any] = response.json()

            progress.progress(80)
            status.write("📊 Preparing results...")

            progress.progress(100)
            status.write("✅ Analysis complete")

        except requests.exceptions.RequestException:
            st.error("❌ Backend not reachable")
            st.stop()

        # ================= RESULTS =================
        st.markdown("### 📊 Analysis Results")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Authenticity", data["prediction"])
        r2.metric("Confidence", f'{data["confidence"]}%')
        r3.metric("Category", data["category"])
        r4.metric("Risk", data["warning"])

        st.markdown("### 🧠 Explanation")
        for reason in data["explanation"]:
            st.write(f"• {reason}")

        st.markdown("---")

        t1, t2 = st.columns(2)
        with t1:
            st.write("**Real Probability**")
            st.progress(data["real_prob"] / 100)
            st.write(f'{data["real_prob"]}%')

        with t2:
            st.write("**Fake Probability**")
            st.progress(data["fake_prob"] / 100)
            st.write(f'{data["fake_prob"]}%')

        st.session_state["analysis_history"].insert(0, {
            "preview": news_text[:50] + "...",
            "prediction": data["prediction"],
            "confidence": float(data["confidence"])
        })

# ================= RIGHT PANEL =================
with col2:
    st.subheader("📤 Upload CSV")

    uploaded = st.file_uploader(
        "Upload CSV with text column",
        type=["csv"]
    )

    if uploaded:
        df = pd.read_csv(uploaded)  # type: ignore
        st.success(f"Loaded {len(df)} rows")

        text_col = None
        for c in ["text", "content", "article", "news", "title"]:
            if c in df.columns:
                text_col = c
                break

        if not text_col:
            st.error("No text column found")
        else:
            sample = st.slider(
                "Articles to analyze",
                1,
                min(20, len(df)),
                5
            )

            if st.button("📊 Analyze Batch", use_container_width=True):
                results: List[Dict[str, Any]] = []
                bar = st.progress(0)

                for i, txt in enumerate(df[text_col].head(sample)):
                    if isinstance(txt, str) and len(txt.strip()) > 30:
                        r = requests.post(
                            PREDICT_URL,
                            json={"text": txt},
                            timeout=30
                        ).json()

                        results.append({
                            "Article": i + 1,
                            "Prediction": r["prediction"],
                            "Confidence": f'{r["confidence"]}%',
                            "Category": r["category"]
                        })

                    bar.progress((i + 1) / sample)

                st.dataframe(pd.DataFrame(results), use_container_width=True)  # type: ignore

# ================= DASHBOARD =================
st.markdown("---")
st.subheader("📊 System Dashboard")

try:
    requests.get(BACKEND_BASE, timeout=3)
    backend_status = "✅ Active"
except requests.exceptions.RequestException:
    backend_status = "❌ Down"

d1, d2, d3, d4 = st.columns(4)
d1.metric("Backend Status", backend_status)
d2.metric("Model", "Logistic Regression")
d3.metric("Features", "TF-IDF (5k)")
d4.metric("Deployment", "Streamlit + FastAPI")

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

• Fake News Detection  
• Confidence Scoring  
• Category Detection  
• Explanation Engine 
• Batch Analysis  

Backend: FastAPI  
Frontend: Streamlit
""")

st.sidebar.markdown("---")
st.sidebar.write("Version: 1.0.0")
st.sidebar.write("Updated: Jan 2026")
