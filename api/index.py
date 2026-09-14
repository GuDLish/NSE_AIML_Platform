from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
import os

app = FastAPI(
    title="NSE Top-10 AI/ML Stock Market Platform",
    description="AI/ML stock analytics API",
    version="1.0.0",
)

# Allow your frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE_PATH = os.path.join(BASE_DIR, "outputs", "app_bundle.json")


@app.get("/")
def home():
    return {
        "status": "online",
        "message": "NSE Top-10 AI/ML Stock Market Platform API",
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy"
    }


@app.get("/api/data")
def get_data():
    if not os.path.exists(BUNDLE_PATH):
        return {
            "error": "app_bundle.json not found"
        }

    with open(BUNDLE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)