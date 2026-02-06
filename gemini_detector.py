import os
import time
import json
import io
from google import genai
from pydantic import BaseModel
from typing import List, Optional
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

class DetectionResult(BaseModel):
    best_match: str
    category: str
    confidence_score: float
    estimated_price: str
    detected_text: str
    description: str
    features: List[str]

class GeminiDetector:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            print("WARNING: No GEMINI_API_KEY provided. Detection will fail until set.")
            self.client = None
        else:
            self.client = genai.Client(api_key=self.api_key)
            print("Gemini API Client initialized.")

    async def detect(self, image_data):
        """
        Detect objects in an image.
        image_data: Can be a file path (str) or raw image bytes.
        """
        if not self.client:
             return {"error": "API Client not configured."}

        is_bytes = isinstance(image_data, (bytes, bytearray))

        if not is_bytes and not os.path.exists(image_data):
            return {"error": f"Image not found at {image_data}"}

        print(f"Sending {'raw bytes' if is_bytes else image_data} to Gemini...")

        try:
            if is_bytes:
                img = Image.open(io.BytesIO(image_data))
            else:
                img = Image.open(image_data)

            with img:
                # Optimized prompt for JSON output
                prompt = """
                Analyze this image and identify the primary object, piece of furniture, appliance, or household item visible.
                Return a structured JSON object with details about the item.
                """
    
                start_time = time.time()

                response = await self.client.aio.models.generate_content(
                    model='gemini-2.5-flash-lite',
                    contents=[prompt, img],
                    config={
                        'response_mime_type': 'application/json',
                        'response_schema': DetectionResult,
                        'thinking_config': {
                            'thinking_budget': 0
                        }
                    }
                )
                elapsed = time.time() - start_time
            
            # The new SDK parses the response automatically if schema is provided
            result = response.parsed

            # Convert Pydantic model to dict for compatibility
            if result:
                result_dict = result.model_dump()
            else:
                # Fallback if parsing failed but response exists
                result_dict = json.loads(response.text)
            
            print(f"Gemini response received in {elapsed:.2f}s")
            print(f"DEBUG RESULT: {json.dumps(result_dict, indent=2)}")

            print(response.usage_metadata)
            
            return result_dict

        except Exception as e:
            print(f"Gemini Error: {e}")
            return {"error": str(e)}

