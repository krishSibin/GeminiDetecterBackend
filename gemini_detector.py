import os
import time
import json
import io
import google.generativeai as genai
from PIL import Image

from dotenv import load_dotenv

class GeminiDetector:
    def __init__(self, api_key=None):

        load_dotenv()
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            print("WARNING: No GEMINI_API_KEY provided. Detection will fail until set.")
        else:
            genai.configure(api_key=self.api_key)
            print("Gemini API configured.")

       
        self.model = genai.GenerativeModel('gemini-flash-latest')

    def detect(self, image_data):
        """
        Detect objects in an image.
        image_data: Can be a file path (str) or raw image bytes.
        """
        if not self.api_key:
             return {"error": "API Key not configured."}

        is_bytes = isinstance(image_data, (bytes, bytearray))

        if not is_bytes and not os.path.exists(image_data):
            return {"error": f"Image not found at {image_data}"}

        print(f"Sending {'raw bytes' if is_bytes else image_data} to Gemini...")
        start_time = time.time()

        try:
            if is_bytes:
                img = Image.open(io.BytesIO(image_data))
            else:
                img = Image.open(image_data)

            with img:
                # Optimized prompt for JSON output
                prompt = """
                Analyze this image and identify the primary object, piece of furniture, appliance, or household item visible.
                Return a pure JSON object (no markdown formatting) with the following structure:
                {
                    "best_match": "Specific name of the item (e.g., 'Wooden Dining Table', 'Dyson V11 Vacuum', 'Stainless Steel Kettle')",
                    "category": "General category (e.g., 'Furniture', 'Appliance', 'Kitchenware')",
                    "confidence_score": 0.95,
                    "estimated_price": "Approximate price in USD (e.g., '$150', '$2,000'). Provide a range if uncertain.",
                    "detected_text": "Any text visible on the item (brand names, model numbers, instructions, labels, logos)",
                    "description": "A concise 1-sentence description of the item's visual state and placement.",
                    "features": ["List", "of", "visible", "key", "features", "or", "materials"]
                }
                If no clear object is found, return {"error": "No object detected"}.
                """
    
                response = self.model.generate_content([prompt, img])
            
            # Robust extraction of JSON from response
            text_response = response.text
            if "```json" in text_response:
                text_response = text_response.split("```json")[1].split("```")[0].strip()
            elif "```" in text_response:
                text_response = text_response.split("```")[1].split("```")[0].strip()
            else:
                text_response = text_response.strip()
            
            result = json.loads(text_response)
            
            elapsed = time.time() - start_time
            print(f"Gemini response received in {elapsed:.2f}s")
            print(f"DEBUG RESULT: {json.dumps(result, indent=2)}")
            
            return result

        except Exception as e:
            print(f"Gemini Error: {e}")
            return {"error": str(e)}

