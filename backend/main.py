from fastapi import FastAPI, HTTPException
from schemas import NewsRequest, PredictionResponse
from services.fake_news_service import predict_news
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="GenAI News Intelligence Backend",
    description="API for detecting fake news using Machine Learning & GenAI",
    version="2.0.0"
)

# Root endpoint
@app.get("/", tags=["Health Check"])
async def health_check():
    return {
        "status": "OK",
        "message": "🚀 Backend is running successfully"
    }

# Prediction endpoint
@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Prediction"])
async def analyze_news(request: NewsRequest):
    try:
        logger.info(f"Received text: {request.text[:50]}...")

        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Input text cannot be empty")

        result = predict_news(request.text)

        logger.info("Prediction successful")
        return result

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error. Please try again later."
        )