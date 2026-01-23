import streamlit as st
import pandas as pd
import os
import sys
from PIL import Image
import io

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

st.set_page_config(
    page_title="GenAI News Intelligence System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #1E88E5, #43A047);
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin: 10px 0;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
    }
    .success-card {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("📰 GenAI News Intelligence System")
st.markdown("---")

if "news_input" not in st.session_state:
    st.session_state.news_input = ""
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

try:
    from src.sentiment import analyze_sentiment
    from src.fake_news import predict_news
    from src.summarizer import TextSummarizer

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📝 Enter News Article")

        news_text = st.text_area(
            "Paste news content here:",
            height=250,
            placeholder="Paste full news article text here...",
            value=st.session_state.news_input,
            key="news_input"
        )

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            analyze_button = st.button(
                "🔍 Analyze Article",
                type="primary",
                use_container_width=True,
                disabled=not news_text.strip()
            )
        with col_btn2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.news_input = ""
                st.rerun()

        if analyze_button and news_text.strip():
            with st.spinner("Analyzing article..."):
                authenticity = predict_news(news_text)
                sentiment = analyze_sentiment(news_text)
                summarizer = TextSummarizer()
                summary = summarizer.summarize(news_text, max_sentences=2)
                keywords = summarizer.get_keywords(news_text, n=8)

            st.markdown("### 📊 Analysis Results")

            result_col1, result_col2, result_col3, result_col4 = st.columns(4)

            with result_col1:
                if authenticity["prediction"] == "REAL":
                    st.success(f"Authenticity: REAL")
                else:
                    st.error(f"Authenticity: FAKE")

            with result_col2:
                st.metric(
                    "Confidence",
                    f"{authenticity['confidence']}%"
                )

            with result_col3:
                st.metric("Category", authenticity["category"])

            with result_col4:
                st.metric("Sentiment", sentiment["sentiment"])

            if authenticity["prediction"] == "FAKE":
                st.markdown(f'<div class="warning-card"><strong>⚠️ {authenticity["warning"]}</strong></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="success-card"><strong>✅ {authenticity["warning"]}</strong></div>', unsafe_allow_html=True)

            st.markdown("### 🧠 Why this result?")
            for reason in authenticity["explanation"]:
                st.write(f"• {reason}")

            st.markdown("---")

            tab1, tab2, tab3, tab4, tab5 = st.tabs(
                ["📈 Details", "📋 Summary", "🔑 Keywords", "📝 Original", "🎭 Sentiment"]
            )

            with tab1:
                st.write("### 📈 Detailed Probabilities")
                col_prob1, col_prob2 = st.columns(2)
                
                with col_prob1:
                    st.write(f"**Real Probability:** {authenticity['real_prob']}%")
                    st.progress(authenticity["real_prob"] / 100)
                
                with col_prob2:
                    st.write(f"**Fake Probability:** {authenticity['fake_prob']}%")
                    st.progress(authenticity["fake_prob"] / 100)

            with tab2:
                st.write("### 📋 Article Summary")
                st.info(summary)

            with tab3:
                st.write("### 🔑 Key Keywords")
                for k in keywords:
                    st.write(f"• {k}")

            with tab4:
                st.write("### 📝 Original Text")
                st.text_area("Original Article", news_text, height=200, disabled=True, key="original_text")

            with tab5:
                st.write("### 🎭 Sentiment Analysis")
                col_sent1, col_sent2 = st.columns(2)
                with col_sent1:
                    st.write(f"**Sentiment:** {sentiment['sentiment']}")
                    st.write(f"**Score:** {sentiment['compound']:.3f}")
                with col_sent2:
                    st.write(f"**Positive:** {sentiment['positive']:.3f}")
                    st.write(f"**Negative:** {sentiment['negative']:.3f}")
                    st.write(f"**Neutral:** {sentiment['neutral']:.3f}")

    with col2:
        st.subheader("📤 Upload Files")

        uploaded_file = st.file_uploader(
            "Upload any file",
            type=None,
            help="Upload any file type - CSV, Excel, TXT, JSON, Images, PDF, etc."
        )

        if uploaded_file:
            file_name = uploaded_file.name
            file_size = uploaded_file.size
            file_type = file_name.split('.')[-1].lower() if '.' in file_name else "unknown"
            
            st.write(f"**File Name:** {file_name}")
            st.write(f"**File Size:** {file_size:,} bytes")
            st.write(f"**File Type:** {file_type}")
            
            if file_type in ['csv', 'xlsx', 'xls']:
                try:
                    if file_type == 'csv':
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    st.success(f"✅ Successfully loaded {len(df)} rows")
                    
                    text_col = None
                    possible_columns = ["text", "Text", "content", "article", "news", "title", "Content", "Article"]
                    
                    for col in possible_columns:
                        if col in df.columns:
                            text_col = col
                            break
                    
                    if text_col:
                        st.success(f"Found text column: '{text_col}'")
                        
                        sample_size = st.slider(
                            "Articles to analyze",
                            1,
                            min(50, len(df)),
                            10
                        )
                        
                        if st.button("📊 Analyze Data", use_container_width=True):
                            with st.spinner(f"Analyzing {sample_size} articles..."):
                                results = []
                                progress_bar = st.progress(0)
                                
                                for i, text in enumerate(df[text_col].head(sample_size)):
                                    if isinstance(text, str) and text.strip():
                                        auth = predict_news(text)
                                        sent = analyze_sentiment(text)
                                        results.append({
                                            "ID": i + 1,
                                            "Preview": text[:50] + "..." if len(text) > 50 else text,
                                            "Authenticity": auth["prediction"],
                                            "Confidence": f"{auth['confidence']}%",
                                            "Category": auth["category"],
                                            "Sentiment": sent["sentiment"]
                                        })
                                    progress_bar.progress((i + 1) / sample_size)
                                
                                if results:
                                    results_df = pd.DataFrame(results)
                                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                                    
                                    fake_count = (results_df['Authenticity'] == 'FAKE').sum()
                                    real_count = (results_df['Authenticity'] == 'REAL').sum()
                                    
                                    col_stats1, col_stats2 = st.columns(2)
                                    with col_stats1:
                                        st.metric("Real Articles", real_count)
                                    with col_stats2:
                                        st.metric("Fake Articles", fake_count)
                                    
                                    if fake_count > 0:
                                        st.warning(f"{fake_count} out of {sample_size} articles appear to be fake ({fake_count/sample_size*100:.1f}%)")
                    else:
                        st.error("No text column found. Available columns:")
                        st.write(list(df.columns))
                        
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                    
            elif file_type in ['txt', 'text']:
                try:
                    text_content = uploaded_file.getvalue().decode("utf-8")
                    st.text_area("File Content", text_content, height=200)
                    
                    if st.button("🔍 Analyze Text File", use_container_width=True):
                        st.session_state.news_input = text_content
                        st.rerun()
                        
                except:
                    text_content = str(uploaded_file.getvalue())
                    st.text_area("File Content", text_content, height=200)
                    
            elif file_type in ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp']:
                try:
                    image = Image.open(uploaded_file)
                    st.image(image, caption=f"Uploaded Image: {file_name}", use_column_width=True)
                    st.info("📸 Image uploaded successfully. Please paste the text content in the main text area for analysis.")
                except:
                    st.error("Could not display image")
                    
            elif file_type == 'json':
                try:
                    df = pd.read_json(uploaded_file)
                    st.dataframe(df.head(), use_container_width=True)
                    st.success(f"✅ Successfully loaded {len(df)} rows")
                except:
                    st.error("Could not parse JSON file")
                    
            elif file_type == 'pdf':
                st.info("📄 PDF file uploaded. Please extract text manually and paste in the main text area.")
                
            else:
                st.info(f"📎 {file_type.upper()} file uploaded. Preview not available for this file type.")

        st.markdown("---")
        st.subheader("⚡ Quick Actions")
        
        if st.button("📝 Test Sample", use_container_width=True):
            sample_text = """Scientists at MIT have made a groundbreaking discovery in quantum computing. 
            The research team has developed a new quantum processor that can solve complex problems 
            in minutes that would take traditional computers thousands of years. This advancement 
            could revolutionize fields from medicine to climate modeling."""
            
            st.session_state.news_input = sample_text
            st.rerun()
        
        if st.button("📊 View Stats", use_container_width=True):
            st.info("Model Accuracy: 98.9%\nTraining Data: 44,898 articles\nFeatures: 5,000")

    st.markdown("---")
    st.subheader("📊 System Dashboard")

    dash_col1, dash_col2, dash_col3, dash_col4 = st.columns(4)
    dash_col1.metric("Model Status", "✅ Active")
    dash_col2.metric("Accuracy", "98.9%")
    dash_col3.metric("Features", "5,000")
    dash_col4.metric("Dataset", "44,898")

    st.markdown("---")
    st.write("### 📈 Recent Activity")
    
    if analyze_button and news_text.strip():
        st.session_state.analysis_history.insert(0, {
            'text': news_text[:50] + "..." if len(news_text) > 50 else news_text,
            'authenticity': authenticity['prediction'],
            'confidence': authenticity['confidence'],
            'sentiment': sentiment['sentiment'],
            'time': pd.Timestamp.now().strftime("%H:%M:%S")
        })
        
        if len(st.session_state.analysis_history) > 5:
            st.session_state.analysis_history = st.session_state.analysis_history[:5]
    
    if st.session_state.analysis_history:
        for item in st.session_state.analysis_history[:3]:
            with st.expander(f"Analysis at {item['time']}: {item['text']}"):
                st.write(f"**Authenticity:** {item['authenticity']}")
                st.write(f"**Confidence:** {item['confidence']}%")
                st.write(f"**Sentiment:** {item['sentiment']}")

except ImportError as e:
    st.error(f"Import Error: {e}")
    st.info("Please install required packages: pip install streamlit pandas scikit-learn textblob")

except Exception as e:
    st.error(f"Application Error: {e}")
    st.info("Please check if model files exist in 'models' folder")

st.sidebar.title("ℹ️ About")
st.sidebar.info("""
**GenAI News Intelligence System**

Features:
• Fake News Detection
• Confidence Scoring
• Explanation Engine
• Category Detection
• Sentiment Analysis
• Text Summarization
• Keyword Extraction
• Batch Processing

**Model Details:**
Algorithm: Logistic Regression
Accuracy: 98.9%
Features: TF-IDF (5000)
Training Data: 44,898 articles
""")

st.sidebar.markdown("---")
st.sidebar.write("### 🚀 Quick Start")

if st.sidebar.button("🔄 Retrain Model", use_container_width=True):
    st.sidebar.info("Run: python notebooks/model_training.py")

if st.sidebar.button("📊 View Report", use_container_width=True):
    st.sidebar.info("Check report.pdf in project folder")

if st.sidebar.button("🧹 Clear History", use_container_width=True):
    if 'analysis_history' in st.session_state:
        st.session_state.analysis_history = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.write("**Version:** 1.0.0")
st.sidebar.write("**Last Updated:** Jan 2026")