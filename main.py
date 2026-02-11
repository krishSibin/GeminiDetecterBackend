from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from typing import List, Optional
import os
import httpx

from gemini_detector import GeminiDetector

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = GeminiDetector()

SERPAPI_KEY = os.environ.get("SERPAPI_KEY")


async def fetch_price_serpapi(query: str) -> Optional[dict]:
    """
    Returns best price info from Google Shopping via SerpAPI.
    Output example:
      {"price": "₹1,299", "title": "...", "source": "...", "link": "..."}
    """
    if not SERPAPI_KEY:
        return None

    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_shopping",
        "q": query,
        "api_key": SERPAPI_KEY,
        # Optional: if you want India results more consistently:
        # "gl": "in",
        # "hl": "en",
    }

    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()

    results = data.get("shopping_results") or []
    if not results:
        return None

    top = results[0]
    return {
        "price": top.get("price"),
        "title": top.get("title"),
        "source": top.get("source"),
        "link": top.get("link"),
    }


@app.get("/")
def read_root():
    return {"message": "Gemini Object Detection API running"}


# --------------------------------------------------
# CAMERA → RAW BYTES (Uint8List)
# --------------------------------------------------
@app.post("/detect")
async def detect_single(request: Request):
    try:
        data = await request.body()
        if not data:
            return JSONResponse(status_code=400, content={"error": "No image bytes received"})

        # Gemini detection (your detector already accepts List[bytes])
        result = await detector.detect_images([data])

        # Decide search query:
        # 1) model_code if present
        # 2) else best_match
        model_code = result.get("model_code")
        search_query = model_code if model_code else result.get("best_match")

        price_info = await fetch_price_serpapi(search_query) if search_query else None
        result["price_info"] = price_info  # can be null

        return result

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# --------------------------------------------------
# GALLERY → SINGLE OR MULTIPLE (MULTIPART)
# --------------------------------------------------
@app.post("/detect/gallery")
async def detect_gallery(files: List[UploadFile] = File(...)):
    try:
        if not files:
            return JSONResponse(status_code=400, content={"error": "No images uploaded"})

        images: List[bytes] = []
        for file in files:
            data = await file.read()
            if data:
                images.append(data)

        if not images:
            return JSONResponse(status_code=400, content={"error": "Uploaded files are empty"})

        # Gemini detection
        result = await detector.detect_images(images)

        # Your rule:
        model_code = result.get("model_code")
        search_query = model_code if model_code else result.get("best_match")

        price_info = await fetch_price_serpapi(search_query) if search_query else None
        result["price_info"] = price_info
        # print(result)

        return result

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    # Render provides the port via the PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
