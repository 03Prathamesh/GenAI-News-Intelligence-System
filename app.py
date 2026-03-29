import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import time
from datetime import datetime

# ================= CONFIG =================
BACKEND = "https://genai-news-intelligence-system.onrender.com"
API = f"{BACKEND}/api/v1/predict"
HEALTH_API = f"{BACKEND}/health"

st.set_page_config(
    page_title="GenAI News Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= STYLE =================
st.markdown("""
<style>
/* Base styling */
body { background: linear-gradient(135deg,#020617,#0f172a); color:white; }
.glass {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 20px;
    margin-bottom:15px;
    transition: transform 0.2s ease;
}
.glass:hover { transform: translateY(-2px); }
.neon-green { color:#22c55e; text-shadow:0 0 5px #22c55e; }
.neon-red { color:#ef4444; text-shadow:0 0 5px #ef4444; }
.neon-yellow { color:#f59e0b; text-shadow:0 0 5px #f59e0b; }
.stButton>button {
    border-radius:10px;
    background: linear-gradient(90deg,#6366f1,#22c55e);
    color:white;
    font-weight:bold;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    transform: scale(1.02);
    box-shadow: 0 0 15px rgba(34,197,94,0.5);
}
/* Sidebar */
.css-1d391kg { background: rgba(0,0,0,0.2); }
/* Metrics */
[data-testid="stMetricValue"] { font-size: 2rem; }
</style>
""", unsafe_allow_html=True)

# ================= HELPER FUNCTIONS =================
def check_backend_health():
    """Check if backend is reachable"""
    try:
        response = requests.get(HEALTH_API, timeout=5)
        return response.status_code == 200
    except:
        return False

def safe_api_call(text):
    """Make API call with error handling"""
    try:
        response = requests.post(API, json={"text": text}, timeout=30)
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to backend. Please check your network."
    except requests.exceptions.Timeout:
        return None, "Backend timed out. Try again later."
    except requests.exceptions.HTTPError as e:
        return None, f"API error: {e.response.status_code}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"

def add_to_history(data):
    """Add analysis to session history with timestamp"""
    data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.history.insert(0, data)
    # Keep only last 20 entries
    st.session_state.history = st.session_state.history[:20]

def clear_history():
    st.session_state.history = []

# ================= STATE =================
if "history" not in st.session_state:
    st.session_state.history = []
if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None

# ================= SIDEBAR =================
with st.sidebar:
    st.title("🧠 GenAI System")
    
    # Backend health indicator
    with st.spinner("Checking backend..."):
        is_backend_up = check_backend_health()
    if is_backend_up:
        st.success("✅ System Active")
    else:
        st.error("❌ Backend Unreachable")
    
    page = st.radio("Navigation", [
        "🏠 Dashboard",
        "📊 Analytics",
        "🌐 Sources",
        "🕒 History"
    ])
    
    st.markdown("---")
    
    # Example texts
    with st.expander("📄 Example News"):
        if st.button("Use 'Fake News' Example"):
            st.session_state.example_text = """BREAKING: Scientists discover that chocolate is the #1 cause of cancer! Major study shows 100% of cancer patients ate chocolate in their lifetime. Big Pharma doesn't want you to know this!"""
            st.rerun()
        if st.button("Use 'Real News' Example"):
            st.session_state.example_text = """NASA's Perseverance rover has successfully collected its first sample of Martian rock. The sample will be returned to Earth by a future mission for analysis. This marks a major milestone in Mars exploration."""
            st.rerun()
    
    # Clear history button
    if st.button("🗑️ Clear History", use_container_width=True):
        clear_history()
        st.success("History cleared!")
        st.rerun()

# ================= DASHBOARD =================
if page == "🏠 Dashboard":
    st.title("🧠 News Intelligence Dashboard")
    
    # Use example text if set
    default_text = st.session_state.get("example_text", "")
    news = st.text_area("Paste News Article", height=250, value=default_text)
    
    col1, col2 = st.columns([3,1])
    with col1:
        analyze_btn = st.button("🚀 Analyze", use_container_width=True)
    with col2:
        if st.button("🔄 Clear", use_container_width=True):
            st.session_state.example_text = ""
            st.rerun()
    
    if analyze_btn:
        if not news.strip():
            st.warning("Please paste a news article to analyze.")
        elif not is_backend_up:
            st.error("Backend is unreachable. Cannot analyze.")
        else:
            with st.spinner("🧠 Analyzing with AI..."):
                # Simulate some processing (optional)
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01) if i < 50 else None  # quick
                    progress_bar.progress(i+1)
                progress_bar.empty()
                
                data, error = safe_api_call(news)
            
            if error:
                st.error(error)
            else:
                st.session_state.last_analysis = data
                add_to_history(data)
                
                final = data.get("final_prediction", "UNKNOWN")
                color = "neon-green" if final=="REAL" else "neon-red" if final=="FAKE" else "neon-yellow"
                
                # Result card
                st.markdown(f"""
                <div class="glass">
                <h2 class="{color}">🔥 {final}</h2>
                <p>{data.get("verification_status", "No verification status")}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics row
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Prediction", data.get("prediction", "N/A"))
                c2.metric("Confidence", f"{data.get('confidence', 0)}%")
                c3.metric("Category", data.get("category", "N/A"))
                c4.metric("Risk Level", data.get("warning", "N/A"))
                
                # Chart
                df = pd.DataFrame({
                    "Type": ["Real","Fake"],
                    "Score": [data.get("real_prob", 0), data.get("fake_prob", 0)]
                })
                fig = px.pie(df, names="Type", values="Score", hole=0.4,
                             color_discrete_sequence=["#22c55e","#ef4444"])
                st.plotly_chart(fig, use_container_width=True)
                
                # Explanation
                st.subheader("🧠 Indicators")
                for r in data.get("explanation", []):
                    st.success(r)
                
                st.subheader("🤖 AI Explanation")
                st.info(data.get("ai_explanation", "No explanation available."))

# ================= ANALYTICS =================
elif page == "📊 Analytics":
    st.title("📊 System Analytics")
    
    if not st.session_state.history:
        st.warning("No analysis data yet. Run some predictions first.")
    else:
        df = pd.DataFrame(st.session_state.history)
        # Ensure required columns exist
        if 'prediction' not in df.columns:
            st.error("History data missing 'prediction' column.")
        else:
            st.write("### Prediction Distribution")
            fig = px.histogram(df, x="prediction", color="prediction",
                               color_discrete_map={"REAL":"#22c55e","FAKE":"#ef4444"})
            st.plotly_chart(fig)
            
            if 'confidence' in df.columns:
                st.write("### Confidence Trend")
                fig2 = px.line(df, y="confidence", title="Confidence over time")
                st.plotly_chart(fig2)
            
            # Additional metrics
            st.write("### Recent Activity")
            recent = df.head(5)[['prediction','confidence','timestamp']] if 'timestamp' in df.columns else df.head(5)
            st.dataframe(recent, use_container_width=True)

# ================= SOURCES =================
elif page == "🌐 Sources":
    st.title("🌐 Fact Check Sources")
    
    if not st.session_state.history:
        st.warning("Run an analysis first to see real-time verification sources.")
    else:
        latest = st.session_state.history[0]
        sources = latest.get("realtime_verification", {}).get("results", [])
        
        if not sources:
            st.info("No verification sources found for the latest analysis.")
        else:
            trusted = [s for s in sources if s.get("trusted")]
            untrusted = [s for s in sources if not s.get("trusted")]
            
            if trusted:
                st.subheader("✅ Trusted Sources")
                for s in trusted:
                    st.success(f"📰 {s.get('title', 'No title')}")
            if untrusted:
                st.subheader("⚠️ Questionable Sources")
                for s in untrusted:
                    st.warning(f"📰 {s.get('title', 'No title')}")

# ================= HISTORY =================
elif page == "🕒 History":
    st.title("🕒 Analysis History")
    
    if not st.session_state.history:
        st.warning("No history available. Analyze some news first.")
    else:
        for idx, item in enumerate(st.session_state.history):
            timestamp = item.get('timestamp', 'No timestamp')
            prediction = item.get('prediction', 'Unknown')
            confidence = item.get('confidence', 0)
            with st.expander(f"{timestamp} - {prediction} ({confidence}%)"):
                st.write(f"**Confidence:** {confidence}%")
                st.write(f"**Category:** {item.get('category', 'N/A')}")
                st.write(f"**Risk Level:** {item.get('warning', 'N/A')}")
                st.write("**AI Explanation:**")
                st.info(item.get("ai_explanation", "No explanation available."))
                if idx == 0 and st.button("🗑️ Delete This Entry", key=f"del_{idx}"):
                    st.session_state.history.pop(idx)
                    st.rerun()