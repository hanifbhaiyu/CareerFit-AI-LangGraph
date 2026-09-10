"""OCR tool: turn an uploaded CV into plain text.

Three input paths are supported, tried in this order:

1. **Digital PDF** - text is extracted directly with `pypdf`. No OCR needed, and
   the result is exact. Most CVs people upload are of this kind.
2. **Image (or scanned PDF) via Tesseract** - the classic OCR engine, run
   locally through `pytesseract`. Free, offline, no API cost.
3. **Image via Gemini vision** - fallback for when Tesseract is not installed on
   the machine, or when it returns near-empty text because the scan is skewed,
   low-contrast, or heavily designed (two-column CVs defeat Tesseract often).

The fallback chain matters for a real deployment: a CV screenshot from a phone
is a genuinely hard OCR target, and a single engine is not reliable enough.
"""

from __future__ import annotations

import base64
import io
import logging

from langchain_core.messages import HumanMessage

from backend.config import get_settings
from backend.core.llm import get_vision_model

logger = logging.getLogger(__name__)

# Below this many characters we assume OCR failed rather than succeeded.
MIN_USABLE_CHARS = 120


class OCRResult:
    """Extracted text plus which engine produced it."""

    def __init__(self, text: str, engine: str, notes: str = "") -> None:
        self.text = text.strip()
        self.engine = engine
        self.notes = notes

    @property
    def is_usable(self) -> bool:
        return len(self.text) >= MIN_USABLE_CHARS

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "engine": self.engine,
            "characters": len(self.text),
            "notes": self.notes,
        }


def extract_text(file_bytes: bytes, filename: str) -> OCRResult:
    """Main entry point: extract text from an uploaded CV file."""
    lowered = filename.lower()

    if lowered.endswith(".pdf"):
        result = _extract_pdf_text(file_bytes)
        if result.is_usable:
            return result
        logger.info("PDF had little embedded text; treating it as a scan.")
        return _ocr_scanned_pdf(file_bytes)

    if lowered.endswith((".txt", ".md")):
        return OCRResult(file_bytes.decode("utf-8", errors="replace"), engine="plain-text")

    return _ocr_image(file_bytes)


# --------------------------------------------------------------------------- #
# PDF handling
# --------------------------------------------------------------------------- #

def _extract_pdf_text(file_bytes: bytes) -> OCRResult:
    """Pull the embedded text layer out of a PDF."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return OCRResult("", engine="none", notes="pypdf is not installed")

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(pages)
        return OCRResult(
            text,
            engine="pypdf-text-layer",
            notes=f"{len(reader.pages)} page(s)",
        )
    except Exception as exc:  # noqa: BLE001 - surface any parser failure to the user
        logger.warning("PDF text extraction failed: %s", exc)
        return OCRResult("", engine="none", notes=str(exc))


def _ocr_scanned_pdf(file_bytes: bytes) -> OCRResult:
    """Rasterise a scanned PDF and OCR the first pages."""
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return OCRResult(
            "",
            engine="none",
            notes=(
                "This PDF has no text layer and pypdfium2 is not installed, so it "
                "cannot be rasterised. Install pypdfium2, or upload a PNG/JPG instead."
            ),
        )

    try:
        pdf = pdfium.PdfDocument(io.BytesIO(file_bytes))
        collected: list[str] = []
        engine_used = "none"

        for page_index in range(min(len(pdf), 3)):  # 3 pages is plenty for a CV
            bitmap = pdf[page_index].render(scale=2.0)  # 2x for legible small type
            image_bytes = _pil_to_png_bytes(bitmap.to_pil())
            page_result = _ocr_image(image_bytes)
            if page_result.text:
                collected.append(page_result.text)
                engine_used = page_result.engine

        return OCRResult("\n\n".join(collected), engine=engine_used, notes="scanned PDF")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Scanned PDF OCR failed: %s", exc)
        return OCRResult("", engine="none", notes=str(exc))


# --------------------------------------------------------------------------- #
# Image handling
# --------------------------------------------------------------------------- #

def _ocr_image(image_bytes: bytes) -> OCRResult:
    """OCR an image, preferring Tesseract and falling back to Gemini vision."""
    settings = get_settings()

    tesseract_result = _run_tesseract(image_bytes)
    if tesseract_result.is_usable:
        return tesseract_result

    if not settings.ocr_fallback_to_vision:
        return tesseract_result

    logger.info(
        "Tesseract produced %d chars; falling back to vision OCR.",
        len(tesseract_result.text),
    )
    return _run_vision_ocr(image_bytes, reason=tesseract_result.notes or "low text yield")


def _run_tesseract(image_bytes: bytes) -> OCRResult:
    """Run the Tesseract engine, with light preprocessing for accuracy."""
    try:
        import pytesseract
        from PIL import Image, ImageOps
    except ImportError:
        return OCRResult("", engine="none", notes="pytesseract/Pillow not installed")

    settings = get_settings()
    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

    try:
        image = Image.open(io.BytesIO(image_bytes))
        # Greyscale plus autocontrast measurably improves recognition on photos
        # of documents, where lighting is uneven.
        image = ImageOps.autocontrast(image.convert("L"))
        text = pytesseract.image_to_string(image)
        return OCRResult(text, engine="tesseract", notes="local OCR engine")
    except Exception as exc:  # noqa: BLE001 - missing binary raises at call time
        logger.info("Tesseract unavailable or failed: %s", exc)
        return OCRResult("", engine="none", notes=f"tesseract error: {exc}")


def _run_vision_ocr(image_bytes: bytes, reason: str = "") -> OCRResult:
    """Use the multimodal model to transcribe an image the OCR engine struggled with."""
    try:
        model = get_vision_model(temperature=0.0)
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        "Transcribe all readable text from this CV image. Preserve the "
                        "reading order and line breaks. Output only the transcribed "
                        "text, with no commentary or markdown fences."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                },
            ]
        )
        response = model.invoke([message])
        return OCRResult(
            str(response.content),
            engine="gemini-vision",
            notes=f"fallback used ({reason})" if reason else "fallback used",
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Vision OCR failed: %s", exc)
        return OCRResult("", engine="none", notes=f"vision OCR error: {exc}")


def _pil_to_png_bytes(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
