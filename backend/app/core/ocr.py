"""
RECALIX OCR Module
==================
Extracts text from images using EasyOCR.
Fully offline after initial model download.

Future optimization: Two-pass with Tesseract first, then EasyOCR refinement.
"""

import numpy as np
from typing import Optional, List
from PIL import Image

from app.config import (
    OCR_ENABLED,
    OCR_BACKEND,
    OCR_LANGUAGES,
    OCR_GPU,
)

# Lazy-load OCR reader (only when first needed)
_reader = None
_reader_lock = None


def get_reader():
    """Get or create OCR reader (lazy singleton)."""
    global _reader, _reader_lock
    
    if not OCR_ENABLED:
        return None
    
    if _reader is not None:
        return _reader
    
    # Thread-safe initialization
    import threading
    if _reader_lock is None:
        _reader_lock = threading.Lock()
    
    with _reader_lock:
        if _reader is not None:  # Double-check after acquiring lock
            return _reader
        
        try:
            if OCR_BACKEND == "easyocr":
                import easyocr
                print(f"Loading EasyOCR model (languages: {OCR_LANGUAGES})...")
                _reader = easyocr.Reader(
                    OCR_LANGUAGES,
                    gpu=OCR_GPU,
                    verbose=False
                )
                print("✅ EasyOCR ready")
            
            elif OCR_BACKEND == "tesseract":
                import pytesseract
                _reader = "tesseract"  # Marker for Tesseract mode
                print("✅ Tesseract ready")
            
            else:
                raise ValueError(f"Unknown OCR backend: {OCR_BACKEND}")
        
        except ImportError as e:
            print(f"⚠️  OCR library not installed: {e}")
            print("Install with: pip install easyocr")
            _reader = None
        except Exception as e:
            print(f"⚠️  OCR initialization failed: {e}")
            _reader = None
    
    return _reader


def extract_text(image: Image.Image) -> str:
    """
    Extract text from PIL Image.
    
    Returns:
        str: Extracted text, space-separated. Empty string if OCR disabled or fails.
    """
    if not OCR_ENABLED:
        return ""
    
    reader = get_reader()
    if reader is None:
        return ""
    
    try:
        if OCR_BACKEND == "easyocr":
            # Convert PIL to numpy array
            img_array = np.array(image)
            
            # EasyOCR returns list of (bbox, text, confidence)
            results = reader.readtext(img_array)
            
            # Extract only text, filter by confidence
            texts = [
                text for (bbox, text, conf) in results
                if conf > 0.3  # Minimum confidence threshold
            ]
            
            return " ".join(texts)
        
        elif OCR_BACKEND == "tesseract":
            import pytesseract
            return pytesseract.image_to_string(image)
        
        else:
            return ""
    
    except Exception as e:
        print(f"OCR extraction failed: {e}")
        return ""


def extract_filename_tokens(filename: str) -> str:
    """
    Extract searchable tokens from filename.
    
    Examples:
        "IMG_2023_birthday_cake.jpg" → "img 2023 birthday cake"
        "invoice-2024-05-10.png" → "invoice 2024 05 10"
    """
    import re
    
    # Remove extension
    name = filename.rsplit(".", 1)[0] if "." in filename else filename
    
    # Replace separators with spaces
    name = re.sub(r'[_\-\.]', ' ', name)
    
    # Lowercase and collapse multiple spaces
    tokens = " ".join(name.lower().split())
    
    return tokens


def detect_document(ocr_text: str, width: int, height: int, file_size: int) -> bool:
    """
    Heuristic: Is this image likely a document?
    
    Criteria:
    - Contains significant amount of text (>50 words)
    - Aspect ratio suggests document layout
    - File size suggests text content
    """
    from app.config import MIN_DOCUMENT_TEXT_LENGTH
    
    if not ocr_text:
        return False
    
    word_count = len(ocr_text.split())
    
    # Must have substantial text
    if word_count < MIN_DOCUMENT_TEXT_LENGTH:
        return False
    
    # Document-like aspect ratio (portrait or landscape A4-ish)
    if width and height:
        aspect = width / height
        # A4 paper: 1.414 (portrait) or 0.707 (landscape)
        if 0.6 < aspect < 1.8:
            return True
    
    return word_count > MIN_DOCUMENT_TEXT_LENGTH


def detect_screenshot(width: int, height: int) -> bool:
    """
    Heuristic: Is this likely a screenshot?
    
    Screenshots tend to be:
    - Very wide (ultrawide monitor)
    - Very tall (phone screenshot)
    - Exactly desktop resolutions (1920x1080, etc.)
    """
    from app.config import SCREENSHOT_ASPECT_MIN, SCREENSHOT_ASPECT_MAX
    
    if not width or not height:
        return False
    
    aspect = width / height
    
    # Very wide or very tall = likely screenshot
    if aspect < SCREENSHOT_ASPECT_MIN or aspect > SCREENSHOT_ASPECT_MAX:
        return True
    
    # Common desktop resolutions
    common_resolutions = {
        (1920, 1080), (2560, 1440), (3840, 2160),  # Desktop
        (1080, 1920), (1440, 2560), (2160, 3840),  # Phone vertical
    }
    
    if (width, height) in common_resolutions:
        return True
    
    return False
