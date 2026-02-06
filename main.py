from fastapi import FastAPI, Request
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from gemini_detector import GeminiDetector

load_dotenv() # Load environment variables from .env file

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini Detector
detector = GeminiDetector()

@app.get("/")
def read_root():
    return {"message": "Object Detection API with Gemini Flash is running!"}

@app.post("/detect-raw")
async def detect_object_raw(request: Request):
    """
    Endpoint for Flutter (Uint8List). Accepts raw bytes in the request body.
    """
    try:
        data = await request.body()
        if not data:
             return JSONResponse(status_code=400, content={"error": "No data received"})
        
        # Run detection directly on bytes
        result = detector.detect(data)
        
        return result

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
 
    if not os.environ.get("GEMINI_API_KEY"):
         print("CRITICAL: GEMINI_API_KEY environment variable is not set.")
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
