from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from schemas import NewsRequest, PredictionResponse
from services.fake_news_service import predict_news
import logging
import time

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------- APP INIT ----------------
app = FastAPI(
    title="GenAI News Intelligence Backend",
    description="AI-powered Fake News Detection using ML + RAG + OpenAI",
    version="3.0.0"
)

# ---------------- CORS (FOR FRONTEND) ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- HEALTH CHECK ----------------
@app.get("/", tags=["Health"])
async def health_check():
    return {
        "status": "OK",
        "message": "🚀 Backend is running successfully"
    }


# ---------------- PREDICTION ENDPOINT ----------------
@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Prediction"])
async def analyze_news(request: NewsRequest):
    start_time = time.time()

    try:
        text = request.text.strip()

        if not text:
            raise HTTPException(
                status_code=400,
                detail="Input text cannot be empty"
            )

        logger.info(f"📥 Received request: {text[:60]}...")

        result = predict_news(text)

        response_time = round(time.time() - start_time, 2)

        logger.info(f"✅ Prediction done in {response_time}s")

        # Optional: attach response time
        result["response_time"] = f"{response_time}s"

        return result

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")

        raise HTTPException(
            status_code=500,
            detail="Internal Server Error. Please try again later."
        )


# ---------------- GLOBAL ERROR HANDLER ----------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"🔥 Unhandled error: {str(exc)}")

    return HTTPException(
        status_code=500,
        detail="Unexpected server error occurred"
    )