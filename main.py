import os
import sys
import logging
import uvicorn
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

# Add root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

# Ensure output directories exist
OUT_DIR = ROOT_DIR / "out"
FILE_UPLOAD_DIR = OUT_DIR / "file_upload"
FILE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
os.environ["FILE_UPLOAD_OUT_DIR"] = str(FILE_UPLOAD_DIR)

# Import services
# Note: These imports must happen after sys.path update
try:
    from extract_service.api_server import extract
    from bid_purification_service.process_bid import process_bid_purification
except ImportError as e:
    print(f"Error importing services: {e}")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Extract Service starting up...")
    logger.info(f"Root Directory: {ROOT_DIR}")
    logger.info(f"File Upload Directory: {FILE_UPLOAD_DIR}")
    yield
    logger.info("Extract Service shutting down...")

app = FastAPI(title="Extract & Purify Service", version="1.0.0", lifespan=lifespan)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request validation
class ExtractRequest(BaseModel):
    files: List[str]
    items: List[str]
    task: Optional[str] = "default"
    information: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None

class PurifyRequest(BaseModel):
    template_path: str
    model_url: str
    model_key: str
    model_id: str
    pdf_urls: List[str]

@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "1.0.0"}

@app.post("/extract")
async def extract_endpoint(payload: ExtractRequest):
    """
    通用文档信息抽取接口
    """
    try:
        # Convert Pydantic model to dict for internal API
        result = await extract(payload.model_dump())
        return result
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/purify")
async def purify_endpoint(payload: PurifyRequest):
    """
    清标助手接口：基于模板对投标文件进行清洗和检查
    """
    try:
        result = await process_bid_purification(
            payload.template_path,
            payload.model_url,
            payload.model_key,
            payload.model_id,
            payload.pdf_urls
        )
        if result is None:
            raise HTTPException(status_code=500, detail="Purification process returned no result")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Purification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Load environment variables from .env if present
    from dotenv import load_dotenv
    load_dotenv()
    
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
