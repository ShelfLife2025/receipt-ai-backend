from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}
from fastapi import File, UploadFile
from typing import List

@app.post("/parse-receipt")
async def parse_receipt(files: List[UploadFile] = File(...)):
    # For now, just return filenames to confirm it's working
    return {"received_files": [file.filename for file in files]}
