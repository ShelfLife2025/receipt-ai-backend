import os
import json
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
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
# Health / root
# -------------------------
@app.get("/")
def root():
    return {"status": "ok"}

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

# -------------------------
# Parsing helper (OpenAI)
# -------------------------
def parse_items_with_openai(ocr_text: str) -> ParsedItems:
    client = openai_client()

    # Use Chat Completions for broad compatibility
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(ocr_text=ocr_text)},
        ],
        temperature=0,
    )

    content = resp.choices[0].message.content if resp.choices else "[]"

    # Extract JSON safely
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Try to find the first JSON array in the content
        start = content.find("[")
        end = content.rfind("]")
        if start != -1 and end != -1 and end > start:
            data = json.loads(content[start : end + 1])
        else:
            raise HTTPException(status_code=502, detail="Model did not return valid JSON.")

    # Validate with Pydantic
    try:
        items = [ParsedItem(**item) for item in data]
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=f"Parsed JSON validation failed: {ve}")

    # Ensure at least something parsed
    if not items:
        raise HTTPException(status_code=204, detail="No items parsed from receipt.")
    return items

# -------------------------
# Upload endpoints
# -------------------------
def _common_scan_logic(upload: UploadFile) -> ParsedItems:
    if upload is None:
        raise HTTPException(status_code=400, detail="No file provided.")
    if not upload.content_type or "image" not in upload.content_type:
        # Still accept; some clients omit a proper type
        pass
    contents = upload.file.read() if hasattr(upload.file, "read") else None
    if contents is None or contents == b"":
        contents = getattr(upload, "spool_max_size", None)  # fallback noop
        contents = contents or b""
    if contents == b"":
        contents = upload.file.read()
    if contents == b"":
        raise HTTPException(status_code=400, detail="Uploaded file was empty.")

    # OCR then parse
    ocr_text = ocr_image_bytes(contents)
    return parse_items_with_openai(ocr_text)

@app.post("/scan", response_model=ParsedItems)
async def scan(
    image: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None),
):
    """
    Accepts multipart/form-data with either field name 'image' or 'file'.
    Returns a JSON array of {name, quantity, category}.
    """
    upload = image or file
    return _common_scan_logic(upload)

# Aliases for safety while the app stabilizes
@app.post("/api/scan", response_model=ParsedItems)
async def scan_api(
    image: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None),
):
    return _common_scan_logic(image or file)

@app.post("/parse", response_model=ParsedItems)
async def parse_alias(
    image: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None),
):
    return _common_scan_logic(image or file)
