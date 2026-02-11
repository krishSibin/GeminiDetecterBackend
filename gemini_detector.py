import os
import time
import json
import io
from google import genai
from pydantic import BaseModel
from typing import List, Union
from PIL import Image
from dotenv import load_dotenv

load_dotenv()


from typing import Optional

class DetectionResult(BaseModel):
    best_match: str
    category: str
    confidence_score: float
    model_code: Optional[str] = None



class GeminiDetector:
    @staticmethod
    def load_image_from_bytes(data: bytes) -> Image.Image:
        """
        Convert raw bytes into a PIL Image in RGB format.
        """
        img = Image.open(io.BytesIO(data))
        img = img.convert("RGB")
        img.load()
        return img

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            print("WARNING: No GEMINI_API_KEY provided. Detection will fail until set.")
            self.client = None
        else:
            self.client = genai.Client(api_key=self.api_key)
            print("Gemini API Client initialized.")

    async def detect(self, image_data: Union[bytes, List[bytes]]):
        """
        Detect objects in single or multiple images.
        image_data: raw bytes (single image) or list of bytes (multiple images)
        """
        if not self.client:
            return {"error": "API Client not configured."}

        # Normalize input to a list
        if isinstance(image_data, (bytes, bytearray)):
            images = [image_data]
        elif isinstance(image_data, list):
            images = image_data
            # Ensure all elements are bytes
            if not all(isinstance(img, (bytes, bytearray)) for img in images):
                return {"error": "All items in the list must be bytes"}
        else:
            return {"error": "Invalid input type. Must be bytes or list of bytes"}

        # Prompt for Gemini API
        prompt = """
        You may receive one or more images of the same item.

        Each image may represent:
        - the product itself
        - a close-up of a label, model number, or product code
        - packaging or documentation

        Your task:
        1. Determine what each image represents.
        2. Identify the product brand and category using visual evidence.
        3. If a model number / model code is visible (e.g., alphanumeric code near the word "Model"),
        extract it and use it to refine the exact product name.
        4. DO NOT return serial numbers. Ignore serial numbers completely.
        5. If both product image and model number are present, prioritize the model number
        for identifying the exact product.
        6. If no model number is visible, identify the product using visual appearance only.
        7. If information is insufficient, return the most accurate partial result
        and lower the confidence score.

        Return structured JSON only with:
        - best_match
        - category
        - confidence_score
        - model_code (null if not found)
        """


        contents = [prompt]

        # Convert each image bytes to PIL Image and append to contents
        for img_bytes in images:
            contents.append(self.load_image_from_bytes(img_bytes))

        try:
            start_time = time.time()

            response = await self.client.aio.models.generate_content(
                model='gemini-2.5-flash-lite',
                contents=contents,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': DetectionResult,
                    'thinking_config': {'thinking_budget': 0}
                }
            )

            elapsed = time.time() - start_time

            # Parse the result using Pydantic model
            result = response.parsed
            result_dict = result.model_dump() if result else json.loads(response.text)

            print(f"Gemini response received in {elapsed:.2f}s")
            print(f"DEBUG RESULT: {json.dumps(result_dict, indent=2)}")
            print(response.usage_metadata)

            return result_dict

        except Exception as e:
            print(f"Gemini Error: {e}")
            return {"error": str(e)}

    async def detect_images(self, images: List[bytes]):
        """
        Helper for FastAPI endpoints to call detect() with multiple images
        """
        return await self.detect(images)
