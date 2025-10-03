import os
import json
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google.cloud import vision
from google.cloud.vision_v1.types.image_annotator import AnnotateImageRequest
from openai import OpenAI

# -------------------------
# App & CORS
# -------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # tighten to your app domains later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Models
# -------------------------
class ParsedItem(BaseModel):
    name: str = Field(..., description="Canonical product name, e.g. 'Chicken Breast'")
    quantity: int = Field(..., ge=1, description="Integer quantity >= 1")
    category: str = Field(..., pattern=r"^(Food|Household)$", description="'Food' or 'Household'")

ParsedItems = List[ParsedItem]

# -------------------------
# Clients (lazy init)
# -------------------------
_vision_client: Optional[vision.ImageAnnotatorClient] = None
_openai_client: Optional[OpenAI] = None

def vision_client() -> vision.ImageAnnotatorClient:
    global _vision_client
    if _vision_client is None:
        # GOOGLE_APPLICATION_CREDENTIALS must be set (Render: Secret File)
        _vision_client = vision.ImageAnnotatorClient()
    return _vision_client

def openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client

# -------------------------
# Health
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -------------------------
# OCR helper
# -------------------------
def ocr_image_bytes(image_bytes: bytes) -> str:
    """
    Uses Google Cloud Vision to extract full text from a receipt image.
    Returns a single string with the OCR text.
    """
    client = vision_client()
    request = AnnotateImageRequest(
        image=vision.Image(content=image_bytes),
        features=[vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)],
    )
    resp = client.annotate_image(request=request)
    if resp.error.message:
        raise RuntimeError(f"Vision API error: {resp.error.message}")
    return resp.full_text_annotation.text if resp.full_text_annotation else ""

# -------------------------
# Prompt for the parser
# -------------------------
SYSTEM_PROMPT = """You are a precise receipt parser. You read raw OCR text from a grocery/retail receipt.
Extract only real purchasable line items. Exclude totals, tax, discounts, store headers/addresses, dates,
barcodes, membership lines, and payment lines.

Return a compact JSON array where each element has:
- name: normalized product name (e.g., "Chicken Breast", "Bananas", "Paper Towels")
- quantity: integer (>=1). If the line implies a pack/weight count like "x2", "2 CT", or a quantity in
  parentheses, use that. If absent, default to 1.
- category: "Food" or "Household" (food/produce/meat/dairy/frozen/snacks = Food; paper goods/cleaners/
  toiletries/laundry/foil/bags/detergent = Household).

Output STRICT JSON only. No comments, no code fences, no extra keys.
"""

USER_PROMPT_TEMPLATE = """OCR_TEXT:
{ocr_text}
"""
