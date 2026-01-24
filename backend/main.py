from fastapi import FastAPI
from schemas import NewsRequest, PredictionResponse
from services.fake_news_service import predict_news

app = FastAPI(
    title="GenAI News Intelligence Backend",
    version="1.0.0"
)

@app.get("/")
def health_check():
    return {"status": "Welcome, Backend running successfully"}

@app.post("/predict", response_model=PredictionResponse)
def analyze_news(request: NewsRequest):
    result = predict_news(request.text)
    return result
