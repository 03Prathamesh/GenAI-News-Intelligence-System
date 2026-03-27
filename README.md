# 📰 GenAI News Intelligence System

## Overview
**GenAI News Intelligence System** is an individual final-year project that uses **Machine Learning and Generative AI**
to analyze news articles. The system detects **fake news**, performs **sentiment analysis**, generates **summaries**,
and provides **confidence scores with explanations** using NLP techniques.

The project is designed for real-world usability, academic evaluation, and placement readiness.

---

## 🚀 Features
- Fake News Detection using Logistic Regression
- Confidence Score & Probability Breakdown
- TF-IDF based Feature Extraction
- Sentiment Analysis (Positive / Neutral / Negative)
- Text Summarization using Generative AI
- Keyword Extraction
- Batch CSV News Analysis
- Interactive Streamlit Web Interface
- Explainable Predictions (Feature Importance)

---

## 🧠 Tech Stack
- **Python 3.10+**
- **Scikit-learn**
- **NLTK**
- **Transformers (Hugging Face)**
- **Streamlit**
- **Pandas & NumPy**

---

## 📁 Project Structure
GenAI-News-Intelligence-System/
│
├── backend/
│   ├── main.py
│   ├── schemas.py
│   ├── requirements.txt
│   ├── services/
│   │   ├── fake_news_service.py
│   │   ├── rag_factcheck.py
│   │   ├── realtime_service.py
│   │   └── openai_explainer.py
│   └── .gitignore
│
├── models/
│   ├── fake_news_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── src/
│   ├── preprocess.py
│   ├── sentiment.py
│   └── summarizer.py
│
├── data/                # (ignored in Git)
├── app.py               # Streamlit UI
├── README.md
└── report.pdf

---

## ⚙️ Installation

pip install -r requirements.txt

▶️ Run the Application
streamlit run app.py