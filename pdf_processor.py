"""
Complete PDF → Image-Question Pipeline Overhaul

This module provides a robust, comprehensive pipeline that:
1. Converts PDF pages to high-resolution images using pdf2image with dynamic pixel scaling
2. Uses PyMuPDF get_text("words") to extract question number positions with strict filtering
3. Groups words into lines with y-coordinate tolerance for better anchor detection
4. Applies strict margin filters and regex patterns for question numbers
5. Falls back to Tesseract OCR if PyMuPDF anchors are insufficient
6. Crops full-width question images with scaled coordinates, minimum heights, and safety margins
7. Saves crops to staging directory with debug overlays
8. Persists to DB via update_or_create with overwrite protection
9. Parses answers from machine-readable text PDFs and updates correct_option field
10. Generates comprehensive pipeline reports with debug information
"""

import os
import re
import json
import shutil
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from django.conf import settings
from django.db import transaction
from django.core.files import File
from django.core.files.base import ContentFile
import io

# Import dependencies with graceful fallback
DEPENDENCIES_AVAILABLE = True
MISSING_DEPS = []

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    MISSING_DEPS.append('Pillow')

try:
    import fitz  # PyMuPDF
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    MISSING_DEPS.append('PyMuPDF')

try:
    from pdf2image import convert_from_path
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    MISSING_DEPS.append('pdf2image')

try:
    import pytesseract
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    MISSING_DEPS.append('pytesseract')

try:
    import cv2
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    MISSING_DEPS.append('opencv-python')

# Configure pytesseract for Windows (only if pytesseract is available)
if 'pytesseract' not in MISSING_DEPS:
    if hasattr(settings, 'TESSERACT_CMD') and os.path.exists(settings.TESSERACT_CMD):
        pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD
    elif os.name == 'nt':  # Windows fallback paths
        possible_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        ]
        for path in possible_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                break

logger = logging.getLogger(__name__)

# Configuration constants
DEDUP_EPS_PTS = 2.0  # Deduplication tolerance in points
MIN_GAP_PX = 80      # Minimum gap between questions in pixels
MIN_HEIGHT_PX = 250  # Minimum crop height in pixels (increased for better option capture)
SAFETY_EXTEND_PX = 200  # Safety extension for short crops (increased)
PADDING_TOP_PX = 8      # Top padding above anchor
PADDING_BOTTOM_PX = 8   # Bottom padding below anchor (reduced for cleaner visuals)
BOTTOM_QUESTION_EXTRA_PX = 40   # Extra padding for bottom questions on page (reduced)
OPTION_SEARCH_EXTEND_PX = 80  # Extended search area for option markers

# Math content detection constants
MATH_CONTENT_EXTRA_TOP_PX = 30    # Extra top padding for math content with superscripts
MATH_CONTENT_EXTRA_BOTTOM_PX = 50  # Extra bottom padding for math content with subscripts
MATRIX_EXTRA_TOP_PX = 60          # Extra top padding for matrix content
MATRIX_EXTRA_BOTTOM_PX = 40       # Extra bottom padding for matrix content
FRACTION_EXTRA_TOP_PX = 25        # Extra top padding for fractions
FRACTION_EXTRA_BOTTOM_PX = 25     # Extra bottom padding for fractions


def create_staging_directory() -> str:
    """Create timestamped staging directory for pipeline artifacts."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    staging_dir = os.path.join(settings.MEDIA_ROOT, 'staging', f'pipeline_{timestamp}')
    os.makedirs(staging_dir, exist_ok=True)
    os.makedirs(os.path.join(staging_dir, 'crops'), exist_ok=True)
    return staging_dir


def render_pdf_pages(pdf_path: str, dpi: int = 300) -> Tuple[List[str], List[Image.Image], List[Tuple[float, float]]]:
    """
    Convert PDF pages to high-resolution images using pdf2image.
    
    Args:
        pdf_path: Path to the PDF file
        dpi: Resolution for image conversion (default 300)
        
    Returns:
        Tuple of (image_paths, PIL_images, scale_factors) for further processing
    """
    try:
        logger.info(f"Converting PDF to images at {dpi} DPI: {pdf_path}")
        images = convert_from_path(pdf_path, dpi=dpi)
        
        # Save images to staging directory
        staging_dir = create_staging_directory()
        image_paths = []
        scale_factors = []
        
        # Open PDF for coordinate scaling
        doc = fitz.open(pdf_path)
        
        for i, image in enumerate(images):
            image_path = os.path.join(staging_dir, f'page_{i+1}.jpg')
            image.save(image_path, 'JPEG', quality=95)
            image_paths.append(image_path)
            
            # Calculate exact scaling factors for this page
            if i < len(doc):
                page = doc.load_page(i)
                scale_x = image.width / page.rect.width
                scale_y = image.height / page.rect.height
                scale_factors.append((scale_x, scale_y))
                logger.debug(f"Page {i+1}: scale_x={scale_x:.3f}, scale_y={scale_y:.3f}")
            
            logger.debug(f"Saved page {i+1} to {image_path}")
        
        doc.close()
        logger.info(f"Successfully converted {len(images)} pages to images")
        return image_paths, images, scale_factors
        
    except Exception as e:
        logger.exception(f"Failed to convert PDF to images: {e}")
        return [], [], []


def extract_number_positions_from_pdf_format1(pdf_path: str) -> List[Tuple[int, int, float]]:
    """
    Extract question number positions from PDF using PyMuPDF with line-aware filtering.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of tuples: (page_index, question_number, y_point)
    """
    try:
        doc = fitz.open(pdf_path)
        positions = []
        
        logger.info(f"Extracting question anchors from PDF using PyMuPDF: {pdf_path}")
        
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            page_height = page.rect.height
            page_width = page.rect.width
            words = page.get_text('words')  # Returns (x0, y0, x1, y1, word, block_no, line_no, word_no)
            
            # Group words into lines by y-coordinate with tolerance
            line_tolerance = 5.0  # Points
            lines = {}
            
            for word_data in words:
                x0, y0, x1, y1, word = word_data[:5]
                
                # Find the closest line for this word
                line_y = None
                for existing_y in lines.keys():
                    if abs(existing_y - y0) <= line_tolerance:
                        line_y = existing_y
                        break
                
                if line_y is None:
                    line_y = y0
                
                if line_y not in lines:
                    lines[line_y] = []
                lines[line_y].append((x0, y0, x1, y1, word))
            
            # Process each line for question numbers
            for line_y, line_words in lines.items():
                # Sort words in line by x-coordinate
                line_words.sort(key=lambda w: w[0])
                
                # Check if first word in line matches question number pattern
                if line_words:
                    first_word = line_words[0]
                    x0, y0, x1, y1, word = first_word
                    
                    # Match question numbers with regex: ^\s*(\d{1,3})\s*[)\.]
                    match = re.match(r'^\s*(\d{1,3})\s*[)\.]', word.strip())
                    if match:
                        question_num = int(match.group(1))
                        
                        # Apply strict margin filters
                        # Accept only tokens within 22% of the left margin
                        if x0 > (page_width * 0.22):
                            continue
                        
                        # Exclude anything in the top 3% but be more lenient for bottom questions
                        # Allow questions closer to bottom edge to catch last questions like Q25
                        if y0 < (page_height * 0.03) or y0 > (page_height * 0.998):
                            continue
                        
                        # Skip tokens within the rightmost 12% (typical page numbers)
                        if x0 > (page_width * 0.88):
                            continue
                        
                        # Skip any line containing a slash / (avoid "1/2", "2/2")
                        line_text = ' '.join([w[4] for w in line_words])
                        if '/' in line_text:
                            continue
                        
                        # Skip lines starting with option markers like (a), (b), (c), (d)
                        if any(re.match(r'^\s*\([a-dA-D]\)', w[4].strip()) for w in line_words):
                            continue
                        
                        positions.append((page_index, question_num, y0))
                        logger.debug(f"Found question {question_num} at page {page_index+1}, y={y0}")
        
        # For each (page, qnum), keep the topmost y and drop near-duplicates
        best_by_key: Dict[Tuple[int, int], float] = {}
        for page_idx, q_num, y_coord in positions:
            key = (page_idx, q_num)
            if key not in best_by_key:
                best_by_key[key] = y_coord
            else:
                # keep the smallest y (topmost)
                if y_coord + DEDUP_EPS_PTS < best_by_key[key]:
                    best_by_key[key] = y_coord
        deduplicated = [(k[0], k[1], y) for k, y in best_by_key.items()]
        deduplicated.sort(key=lambda x: (x[0], x[2]))
        
        # Enhanced validation for missing questions, especially on final pages
        if deduplicated:
            # Check for potential missing questions on the last page
            last_page_anchors = [anchor for anchor in deduplicated if anchor[0] == len(doc) - 1]
            if len(last_page_anchors) == 0 and len(doc) > 1:
                # No questions found on last page, do a more aggressive search
                logger.warning("No questions found on last page, performing aggressive search")
                last_page = doc.load_page(len(doc) - 1)
                last_page_words = last_page.get_text('words')
                
                # Look for any number pattern that could be a question
                for word_data in last_page_words:
                    x0, y0, x1, y1, word = word_data[:5]
                    # More relaxed pattern matching for last page
                    match = re.match(r'^\s*(\d{1,3})\s*[)\.]?', word.strip())
                    if match:
                        question_num = int(match.group(1))
                        # More lenient position checks for last page
                        if (x0 <= (last_page.rect.width * 0.3) and 
                            y0 >= (last_page.rect.height * 0.02) and 
                            y0 <= (last_page.rect.height * 0.998)):
                            deduplicated.append((len(doc) - 1, question_num, y0))
                            logger.info(f"Found missing question {question_num} on last page with aggressive search")
                            break
        
        # Re-sort after potential additions
        deduplicated.sort(key=lambda x: (x[0], x[2]))
        
        logger.info(f"Found {len(deduplicated)} question anchors using PyMuPDF (after deduplication and validation)")
        return deduplicated
        
    except Exception as e:
        logger.exception(f"Failed to extract anchors with PyMuPDF: {e}")
        return []


def extract_number_positions_from_pdf_format2(pdf_path: str) -> List[Tuple[int, int, float]]:
    """
    Extract question number positions from PDF using Format 2 parsing rules.
    Improvements over Format 1:
    - Span-level parsing with font size heuristics (prefers larger/heading-like spans)
    - Stricter left-margin requirement and page margin guards
    - Reject '0.' and implausible numbers
    - Enforce minimal vertical distance between anchors to avoid mid-line artifacts
    """
    try:
        doc = fitz.open(pdf_path)
        positions: List[Tuple[int, int, float]] = []

        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            page_height = page.rect.height
            page_width = page.rect.width

            # Use dict to get spans with font size
            pdict = page.get_text("dict")

            # Collect candidate anchors from spans at line starts near left margin
            candidates: List[Tuple[int, float, float, float]] = []  # (qnum, y0, x0, fontsize)
            for block in pdict.get("blocks", []):
                for line in block.get("lines", []):
                    if not line.get("spans"):
                        continue
                    # Consider first non-empty span on the line
                    first_span = None
                    for sp in line["spans"]:
                        text = (sp.get("text") or "").strip()
                        if text:
                            first_span = sp
                            break
                    if not first_span:
                        continue
                    text = first_span.get("text", "").strip()
                    x0 = first_span.get("origin", [0, 0])[0] if first_span.get("origin") else line.get("bbox", [0, 0, 0, 0])[0]
                    y0 = first_span.get("origin", [0, 0])[1] if first_span.get("origin") else line.get("bbox", [0, 0, 0, 0])[1]
                    size = float(first_span.get("size", 0.0) or 0.0)

                    # Left margin & vertical page margin guards
                    if x0 > (page_width * 0.18):
                        continue
                    if y0 < (page_height * 0.03) or y0 > (page_height * 0.995):
                        continue

                    # Skip option marker lines and slash lines entirely
                    full_line_text = "".join((sp.get("text") or "") for sp in line.get("spans", []))
                    if re.search(r"\(\s*[A-Da-d]\s*\)|\b[ABCD][\).]", full_line_text):
                        continue
                    if "/" in full_line_text:
                        continue

                    # Match number patterns at start
                    m = re.match(r"^(\d{1,3})\s*[\)\.]\s*$", text)
                    if not m:
                        # Some PDFs split tokens: try first two spans concatenated
                        if len(line.get("spans", [])) >= 2:
                            tcat = (line["spans"][0].get("text") or "") + (line["spans"][1].get("text") or "")
                            tcat = tcat.strip()
                            m = re.match(r"^(\d{1,3})\s*[\)\.]", tcat)
                        if not m:
                            continue

                    qn = int(m.group(1))
                    if qn <= 0 or qn > 400:
                        continue

                    # Heuristic: font size should be in the top 60% of sizes seen on page (avoid tiny mid-text '0.')
                    # Gather a quick distribution of sizes per page only once
                    candidates.append((qn, float(y0), float(x0), size))

            # If we have sizes, compute a cutoff
            if candidates:
                sizes = [c[3] for c in candidates]
                try:
                    sizes_sorted = sorted(sizes)
                    cutoff = sizes_sorted[max(0, int(len(sizes_sorted) * 0.4))]  # keep spans >= 40th percentile
                except Exception:
                    cutoff = 0.0
            else:
                cutoff = 0.0

            for qn, y0, x0, size in candidates:
                if size < cutoff:
                    continue
                positions.append((page_index, qn, y0))

        # Deduplicate and order
        by_key: Dict[Tuple[int, int], float] = {}
        for p, q, y in positions:
            k = (p, q)
            if k not in by_key or y + DEDUP_EPS_PTS < by_key[k]:
                by_key[k] = y
        dedup = [(k[0], k[1], y) for k, y in by_key.items()]
        dedup.sort(key=lambda t: (t[0], t[2]))

        # Enforce minimal spacing between anchors to remove mid-line artifacts
        filtered: List[Tuple[int, int, float]] = []
        last_y_by_page: Dict[int, float] = {}
        for p, q, y in dedup:
            ly = last_y_by_page.get(p)
            if ly is not None and (y - ly) < 16.0:  # 16pt ~ 5.6mm
                continue
            last_y_by_page[p] = y
            filtered.append((p, q, y))

        logger.info(f"[Format2] Anchors found (filtered): {len(filtered)}")
        return filtered
    except Exception:
        logger.exception("Format 2 anchor extraction failed")
        return []


def extract_number_positions_from_pdf(pdf_path: str) -> List[Tuple[int, int, float]]:
    """
    Extract question number positions from PDF using PyMuPDF with line-aware filtering.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of tuples: (page_index, question_number, y_point)
    """
    try:
        format1_positions = extract_number_positions_from_pdf_format1(pdf_path)
        format2_positions = extract_number_positions_from_pdf_format2(pdf_path)
        positions = format1_positions + format2_positions
        return positions
        
    except Exception as e:
        logger.exception(f"Failed to extract anchors with PyMuPDF: {e}")
        return []


def detect_math_content_in_region(page, top_pt: float, bottom_pt: float, subject: str = None) -> Dict[str, bool]:
    """
    Detect mathematical content in a specific region of the page.
    
    Args:
        page: PyMuPDF page object
        top_pt: Top boundary in PDF coordinates
        bottom_pt: Bottom boundary in PDF coordinates
        subject: Subject name for enhanced detection
        
    Returns:
        Dictionary with detected math content types
    """
    try:
        # Get text with detailed positioning
        words = page.get_text("words")
        text_blocks = page.get_text("dict")
        
        math_indicators = {
            'has_superscript': False,
            'has_subscript': False,
            'has_fractions': False,
            'has_matrices': False,
            'has_complex_math': False,
            'is_math_subject': subject and 'math' in subject.lower()
        }
        
        # Patterns for mathematical content
        superscript_patterns = [r'\^\d+', r'\^\{.*?\}', r'\^\(.*?\)', r'²', r'³', r'⁴', r'⁵', r'⁶', r'⁷', r'⁸', r'⁹']
        subscript_patterns = [r'_\d+', r'_{.*?}', r'_\(.*?\)', r'₀', r'₁', r'₂', r'₃', r'₄', r'₅', r'₆', r'₇', r'₈', r'₉']
        fraction_patterns = [r'\\frac\{.*?\}\{.*?\}', r'\d+/\d+', r'[a-zA-Z]+/[a-zA-Z]+', r'\([^)]+\)/\([^)]+\)']
        matrix_patterns = [r'\\begin\{.*?matrix\}', r'\[.*?\]', r'\|.*?\|', r'det\(', r'matrix']
        
        # Check words in the region
        region_text = ""
        for word in words:
            word_y = float(word[1])
            if top_pt <= word_y <= bottom_pt:
                word_text = str(word[4])
                region_text += word_text + " "
                
                # Check for mathematical symbols and patterns
                for pattern in superscript_patterns:
                    if re.search(pattern, word_text, re.IGNORECASE):
                        math_indicators['has_superscript'] = True
                        
                for pattern in subscript_patterns:
                    if re.search(pattern, word_text, re.IGNORECASE):
                        math_indicators['has_subscript'] = True
                        
                for pattern in fraction_patterns:
                    if re.search(pattern, word_text, re.IGNORECASE):
                        math_indicators['has_fractions'] = True
                        
                for pattern in matrix_patterns:
                    if re.search(pattern, word_text, re.IGNORECASE):
                        math_indicators['has_matrices'] = True
        
        # Check text blocks for font size variations (indicating super/subscripts)
        for block in text_blocks.get('blocks', []):
            if 'lines' in block:
                for line in block['lines']:
                    line_y = line['bbox'][1]
                    if top_pt <= line_y <= bottom_pt:
                        font_sizes = []
                        for span in line.get('spans', []):
                            font_sizes.append(span.get('size', 12))
                        
                        # If we have significant font size variations, likely math content
                        if len(set(font_sizes)) > 1:
                            size_diff = max(font_sizes) - min(font_sizes)
                            if size_diff > 2:  # Significant size difference
                                math_indicators['has_complex_math'] = True
        
        # Additional checks for common mathematical expressions
        math_symbols = ['∑', '∫', '∂', '√', '∞', '±', '≤', '≥', '≠', '≈', '∝', '∈', '∉', '⊂', '⊃', '∪', '∩']
        for symbol in math_symbols:
            if symbol in region_text:
                math_indicators['has_complex_math'] = True
                break
        
        # Check for matrix-like structures (numbers in brackets/parentheses)
        if re.search(r'\[\s*[-+]?\d+\s+[-+]?\d+\s*\]', region_text) or \
           re.search(r'\[\s*[-+]?\d+\s*\]', region_text):
            math_indicators['has_matrices'] = True
            
        logger.debug(f"Math content detected in region: {math_indicators}")
        return math_indicators
        
    except Exception as e:
        logger.debug(f"Error detecting math content: {e}")
        return {'has_superscript': False, 'has_subscript': False, 'has_fractions': False, 
                'has_matrices': False, 'has_complex_math': False, 'is_math_subject': False}


def ocr_find_numbers_on_image(image: Image.Image) -> List[Tuple[int, int, int]]:
    """
    Fallback OCR-based question number detection using Tesseract with margin filtering.
    
    Args:
        image: PIL Image object
        
    Returns:
        List of tuples: (question_number, x_pixel, y_pixel)
    """
    try:
        img_width, img_height = image.size
        
        # Use Tesseract to get word-level data
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        positions = []
        for i, word in enumerate(data['text']):
            if not word.strip():
                continue
            
            # Match question numbers with regex
            match = re.match(r'^(\d{1,3})(?:[)\.]?)$', word.strip())
            if match:
                question_num = int(match.group(1))
                
                # Get bounding box
                x = data['left'][i]
                y = data['top'][i]
                w = data['width'][i]
                h = data['height'][i]
                
                # Apply strict margin filters (same as PyMuPDF)
                # Accept only tokens within 22% of the left margin
                if x > (img_width * 0.22):
                    continue
                
                # Exclude anything in the top 3% or bottom 5% of page height
                if y < (img_height * 0.03) or y > (img_height * 0.95):
                    continue
                
                # Skip tokens within the rightmost 12%
                if x > (img_width * 0.88):
                    continue
                
                # Skip any token on a line containing a slash
                line_words = [data['text'][j] for j in range(len(data['text'])) 
                             if abs(data['top'][j] - y) < 10]  # Same line
                if any('/' in w for w in line_words if w.strip()):
                    continue
                
                # Skip lines starting with option markers
                if any(re.match(r'^\s*\([a-dA-D]\)', w.strip()) for w in line_words if w.strip()):
                    continue
                
                positions.append((question_num, x, y))
                logger.debug(f"OCR found question {question_num} at x={x}, y={y}")
        
        return positions
        
    except Exception as e:
        logger.exception(f"Failed OCR fallback anchor detection: {e}")
        return []


def crop_questions_from_images(
    pdf_path: str, 
    image_paths: List[str], 
    pil_images: List[Image.Image],
    anchors: List[Tuple[int, int, float]],
    staging_dir: str,
    overwrite: bool = False,
    validate_mode: str = 'format1',
    subject: str = None
) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]]]:
    """
    Crop question images using look-ahead, safe-crop algorithm with dynamic scaling.
    
    Args:
        pdf_path: Path to PDF for coordinate scaling
        image_paths: List of page image paths
        pil_images: List of PIL Image objects
        anchors: List of (page_index, question_number, y_coordinate) tuples
        staging_dir: Directory for saving crops and debug images
        overwrite: Whether to overwrite existing good crops
        
    Returns:
        List of (question_number, crop_path) tuples
    """
    try:
        logger.info("Starting question cropping with look-ahead algorithm")
        crops: List[Tuple[int, str]] = []
        skipped: List[Tuple[int, str]] = []
        
        # Open PDF for coordinate scaling
        doc = fitz.open(pdf_path)
        
        # Group anchors by page
        page_anchors = {}
        for page_idx, q_num, y_coord in anchors:
            if page_idx not in page_anchors:
                page_anchors[page_idx] = []
            page_anchors[page_idx].append((q_num, y_coord))
        
        for page_idx, page_anchors_list in page_anchors.items():
            if page_idx >= len(pil_images):
                continue
                
            image = pil_images[page_idx]
            img_width, img_height = image.size
            
            # Get PDF page for scaling calculation
            page = doc.load_page(page_idx)
            pdf_height = page.rect.height
            
            # Calculate dynamic vertical scale factor
            scale_y = img_height / pdf_height
            logger.debug(f"Page {page_idx+1}: scale_y = {scale_y} (img_h={img_height}, pdf_h={pdf_height})")
            
            # Sort anchors by y-coordinate
            page_anchors_list.sort(key=lambda x: x[1])
            
            # Create debug overlay image
            debug_image = image.copy()
            draw = ImageDraw.Draw(debug_image)
            
            for i, (q_num, y_coord) in enumerate(page_anchors_list):
                # Detect mathematical content in the question region for enhanced cropping
                # First, get a preliminary region to analyze
                preliminary_bottom = y_coord + 200 / scale_y  # Look ahead ~200px for math detection
                if i + 1 < len(page_anchors_list):
                    next_y_coord = page_anchors_list[i + 1][1]
                    preliminary_bottom = min(preliminary_bottom, next_y_coord)
                
                math_content = detect_math_content_in_region(page, y_coord, preliminary_bottom, subject)
                
                # Calculate enhanced padding based on detected math content
                top_padding = PADDING_TOP_PX
                if math_content['has_superscript'] or math_content['has_complex_math']:
                    top_padding += MATH_CONTENT_EXTRA_TOP_PX
                if math_content['has_matrices']:
                    top_padding += MATRIX_EXTRA_TOP_PX
                if math_content['has_fractions']:
                    top_padding += FRACTION_EXTRA_TOP_PX
                
                # Convert PDF y-coordinate to pixel y-coordinate with enhanced padding
                top_px = int(y_coord * scale_y) - top_padding
                
                logger.debug(f"Q{q_num}: Math content detected: {math_content}, using top_padding: {top_padding}px")
                
                # Determine bottom boundary using look-ahead
                if i + 1 < len(page_anchors_list):
                    next_y_coord = page_anchors_list[i + 1][1]
                    next_top_px = int(next_y_coord * scale_y)
                    
                    gap_px = next_top_px - top_px
                    
                    # Check if gap is sufficient for safe cropping
                    if gap_px >= MIN_GAP_PX:
                        # Calculate enhanced bottom padding based on math content
                        bottom_padding = PADDING_BOTTOM_PX
                        if math_content['has_subscript'] or math_content['has_complex_math']:
                            bottom_padding += MATH_CONTENT_EXTRA_BOTTOM_PX
                        if math_content['has_matrices']:
                            bottom_padding += MATRIX_EXTRA_BOTTOM_PX
                        if math_content['has_fractions']:
                            bottom_padding += FRACTION_EXTRA_BOTTOM_PX
                        
                        bottom_px = next_top_px - bottom_padding
                        logger.debug(f"Q{q_num}: Using enhanced bottom padding: {bottom_padding}px for math content")
                    else:
                        # Gap too small, extend with safety margin
                        bottom_px = top_px + MIN_HEIGHT_PX + SAFETY_EXTEND_PX
                        logger.debug(f"Question {q_num}: small gap ({gap_px}px), extending to {bottom_px - top_px}px")
                else:
                    # Last anchor on page: determine bottom using comprehensive option detection
                    words = page.get_text("words")  # (x0, y0, x1, y1, word, block_no, line_no, word_no)
                    # Accept variants: (A)  A)  A.  [A]  A- (with optional spaces)
                    option_token_regex = re.compile(r"^[\(\[]?\s*([A-Da-d])\s*[\)\]\.-]?$")
                    option_y_points: List[float] = []
                    option_letters_found: List[str] = []
                    
                    # Search in extended area below question anchor for better option detection
                    search_bottom_pt = (y_coord + OPTION_SEARCH_EXTEND_PX / scale_y)
                    
                    for w in words:
                        word_y = float(w[1])
                        # Only consider words below the question anchor and within search area
                        if word_y >= y_coord and word_y <= search_bottom_pt:
                            token = str(w[4]).strip()
                            m = option_token_regex.match(token)
                            if m:
                                option_y_points.append(word_y)
                                letter = m.group(1).upper()
                                if letter in ['A','B','C','D'] and letter not in option_letters_found:
                                    option_letters_found.append(letter)
                    
                    if option_y_points:
                        max_option_y_pt = max(option_y_points)
                        
                        # Calculate enhanced bottom padding based on math content
                        bottom_extra = BOTTOM_QUESTION_EXTRA_PX
                        if math_content['has_subscript'] or math_content['has_complex_math']:
                            bottom_extra += MATH_CONTENT_EXTRA_BOTTOM_PX
                        if math_content['has_matrices']:
                            bottom_extra += MATRIX_EXTRA_BOTTOM_PX
                        if math_content['has_fractions']:
                            bottom_extra += FRACTION_EXTRA_BOTTOM_PX
                        
                        # Apply enhanced padding for math content
                        bottom_px = int(max_option_y_pt * scale_y) + bottom_extra
                        bottom_px = min(bottom_px, img_height - 3)  # Minimal margin for cleaner look
                        
                        # Enhanced logic: if less than 4 options found or this is last page, be more generous
                        is_last_page = (page_idx == len(pil_images) - 1)
                        options_found = len(set(option_letters_found))
                        
                        if options_found < 4:
                            # For incomplete option detection, extend conservatively with math-aware padding
                            bottom_px = max(bottom_px, int(max_option_y_pt * scale_y) + bottom_extra * 1.2)
                            bottom_px = min(bottom_px, img_height - 3)  # Minimal margin for clean appearance
                            logger.debug(f"Q{q_num}: Extended bottom for incomplete options ({options_found}/4) with math padding: {bottom_extra}px")
                        elif is_last_page:
                            # For last page questions, use minimal margin for clean appearance
                            bottom_px = min(bottom_px, img_height - 3)
                            logger.debug(f"Q{q_num}: Last page question, using enhanced bottom boundary with math padding: {bottom_extra}px")
                        else:
                            logger.debug(f"Q{q_num}: Applied enhanced math content padding: {bottom_extra}px")
                    else:
                        # Fallback: go to page bottom with minimal margin for clean visuals
                        bottom_px = img_height - 3
                        logger.debug(f"Q{q_num}: No options found, using page bottom")

                    # Disable page merge to prevent stitching - each question should be standalone
                    # This ensures clean, individual question frames without unwanted content
                    need_merge_next = False  # Completely disable merging to prevent stitching
                
                # Ensure minimum height
                height = bottom_px - top_px
                if height < MIN_HEIGHT_PX:
                    bottom_px = min(img_height, top_px + MIN_HEIGHT_PX + SAFETY_EXTEND_PX)
                    height = bottom_px - top_px
                    logger.debug(f"Extended question {q_num} height to {height}px (was below {MIN_HEIGHT_PX}px threshold)")
                
                # Defensive: ensure bottom_px is not less than or equal to top_px
                if bottom_px <= top_px:
                    bottom_px = min(img_height, top_px + MIN_HEIGHT_PX + SAFETY_EXTEND_PX)
                    logger.debug(f"Adjusted bottom for question {q_num} because bottom_px <= top_px")
                
                # Crop full-width image and save to staging only
                crop_box = (0, top_px, img_width, bottom_px)
                cropped_image = image.crop(crop_box)

                # Page merge functionality disabled to prevent stitching
                # Each question image will be standalone for clean visual appearance
                # (Merge logic removed to ensure no unwanted content is included)
                
                # Validate options presence using text layer within crop region
                try:
                    top_pt = top_px / scale_y
                    bottom_pt = bottom_px / scale_y
                    words = page.get_text("words")
                    # Accept variants: (A)  A)  A.  [A]  A-  and tolerate spacing; also handle single-letter tokens
                    option_token_regex = re.compile(r"^[\(\[]?\s*([A-Da-d])\s*[\)\]\.-]?$")
                    letters_in_block: List[str] = []
                    for w in words:
                        x0 = float(w[0]); y0 = float(w[1])
                        token = str(w[4]).strip()
                        if y0 >= top_pt and y0 <= bottom_pt:
                            m = option_token_regex.match(token)
                            # Do not limit to left-half; options may be in two columns across the page
                            if m:
                                letter = m.group(1).upper()
                                if letter in ['A','B','C','D'] and letter not in letters_in_block:
                                    letters_in_block.append(letter)
                    # Page merge validation removed - each question validated independently
                    # This prevents false validation from next page content
                    # After scanning both current and (possibly) next page top slice, evaluate presence
                    has_all = set(letters_in_block) >= {'A','B','C','D'}
                except Exception:
                    logger.debug("Validation step failed; proceeding without validation", exc_info=True)
                    has_all = True

                if not has_all:
                    # If this is the last page or the last anchor on the page, be lenient and proceed with a warning
                    is_last_page = (page_idx == (len(pil_images) - 1))
                    is_last_anchor_on_page = (i == len(page_anchors_list) - 1)
                    if validate_mode == 'format2' or is_last_page or is_last_anchor_on_page:
                        logger.warning(f"Proceeding without all options detected for Q{q_num}. Found: {','.join(sorted(letters_in_block)) or 'none'}")
                    else:
                        reason = f"Missing options in block: found {','.join(sorted(letters_in_block)) or 'none'}"
                        skipped.append((q_num, reason))
                        logger.warning(f"Skipping Q{q_num}: {reason}")
                        continue

                # Save to staging directory first
                staging_crop_path = os.path.join(staging_dir, 'crops', f'question_{q_num}.jpg')
                cropped_image.save(staging_crop_path, 'JPEG', quality=95)
                
                # Record staging crop path for DB persistence step
                crops.append((q_num, staging_crop_path))

                # Save per-question debug overlay image
                try:
                    per_q_dbg = image.copy()
                    per_q_draw = ImageDraw.Draw(per_q_dbg)
                    per_q_draw.rectangle([0, top_px, img_width, bottom_px], outline='lime', width=3)
                    per_q_draw.text((10, top_px + 5), f'Q{q_num}', fill='lime')
                    per_q_path = os.path.join(staging_dir, f'page_{page_idx+1}_q{q_num}_debug.jpg')
                    per_q_dbg.save(per_q_path, 'JPEG', quality=90)
                except Exception:
                    logger.debug("Failed to save per-question debug overlay", exc_info=True)
                
                # Draw debug rectangle and label on overlay image
                draw.rectangle([0, top_px, img_width, bottom_px], outline='red', width=3)
                draw.text((10, top_px + 5), f'Q{q_num}', fill='red')
                
                logger.debug(f"Cropped question {q_num}: {crop_box} -> {staging_crop_path}")
            
            # Save debug overlay image
            debug_path = os.path.join(staging_dir, f'page_{page_idx+1}_debug.jpg')
            debug_image.save(debug_path, 'JPEG', quality=95)
        
        doc.close()
        logger.info(f"Successfully cropped {len(crops)} questions; skipped {len(skipped)}")
        return crops, skipped
        
    except Exception as e:
        logger.exception(f"Failed to crop questions: {e}")
        return [], []


def save_question_images_to_db(crops: List[Tuple[int, str]], course=None, overwrite: bool = False, subject: Optional[str] = None) -> Dict[str, int]:
    """
    Save question crops to database atomically using update_or_create.
    
    Args:
        crops: List of (question_number, crop_path) tuples
        course: Course instance to associate questions with
        overwrite: Whether to overwrite existing good images
        
    Returns:
        Dictionary with creation/update counts
    """
    try:
        # Import Question inside the function to avoid circular imports
        from exam.models import Question
        
        logger.info(f"Saving {len(crops)} question crops to database")
        created_count = 0
        updated_count = 0
        skipped_count = 0
        
        with transaction.atomic():
            for question_num, crop_path in crops:
                # Check if existing image is good (unless overwrite=True)
                if not overwrite:
                    try:
                        existing_question = Question.objects.filter(
                            question_number=question_num,
                            course=course,
                            subject=subject or 'General'
                        ).first()
                        
                        if existing_question and existing_question.image:
                            # Check if existing image meets quality criteria
                            if os.path.exists(existing_question.image.path):
                                existing_img = Image.open(existing_question.image.path)
                                if existing_img.height >= MIN_HEIGHT_PX:
                                    logger.info(f"Skipping question {question_num}: existing image is good ({existing_img.height}px)")
                                    skipped_count += 1
                                    continue
                    except Exception:
                        # If we can't check existing image, proceed to create/update
                        logger.debug(f"Could not check existing image for question {question_num}, proceeding")
                
                # Open image file from staging path as Django File
                # Convert to PNG and store with .png extension to meet storage requirements
                with Image.open(crop_path) as img_src:
                    buffer = io.BytesIO()
                    img_src.save(buffer, format='PNG')
                    django_file = ContentFile(buffer.getvalue(), name=f'question_{question_num}.png')
                    
                    # Use update_or_create for atomic operation
                    defaults = {
                        'question': f'Question {question_num}',
                        'marks': 1,
                        'image': django_file,
                    }
                    if subject:
                        defaults['subject'] = subject
                    question, created = Question.objects.update_or_create(
                        question_number=question_num,
                        course=course,
                        subject=subject or 'General',
                        defaults=defaults
                    )
                    
                    if created:
                        created_count += 1
                        logger.debug(f"Created question {question_num}")
                    else:
                        updated_count += 1
                        logger.debug(f"Updated question {question_num}")
        
        logger.info(f"Database save completed: {created_count} created, {updated_count} updated, {skipped_count} skipped")
        return {
            'created': created_count, 
            'updated': updated_count, 
            'skipped': skipped_count
        }
        
    except Exception as e:
        logger.exception(f"Failed to save crops to database: {e}")
        return {'created': 0, 'updated': 0, 'skipped': 0}


def parse_answers_pdf(answers_pdf_path: str) -> Dict[int, str]:
    """
    Parse answers from machine-readable PDF using PyMuPDF text extraction.
    
    Args:
        answers_pdf_path: Path to the answers PDF file
        
    Returns:
        Dictionary mapping question numbers to correct answers (A, B, C, D)
    """
    try:
        logger.info(f"Parsing answers from PDF: {answers_pdf_path}")
        doc = fitz.open(answers_pdf_path)
        answers = {}
        
        # Extract all text from PDF
        full_text = ""
        for page in doc:
            full_text += page.get_text("text") + "\n"
        
        # Use regex to extract (question_number, option) pairs
        # Patterns like: "1. A", "2) B", "Question 3: C", etc.
        patterns = [
            r'(\d{1,3})\s*[)\.\-]?\s*([A-Da-d])',
            r'(?:Question\s*)?(\d{1,3})[\s\.\)\:]+([ABCD])',
            r'Q[\s\.]?(\d{1,3})[\s\.\)\:]+([ABCD])',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                q_num = int(match[0])
                option = match[1].upper()
                if option in ['A', 'B', 'C', 'D']:
                    answers[q_num] = option
                    logger.debug(f"Found answer: Question {q_num} = {option}")
        
        doc.close()
        logger.info(f"Parsed {len(answers)} answers from PDF")
        return answers
        
    except Exception as e:
        logger.exception(f"Failed to parse answers PDF: {e}")
        return {}


def refine_anchors_format2(pdf_path: str, anchors: List[Tuple[int, int, float]]) -> List[Tuple[int, int, float]]:
    """Refine Format 2 anchors by scanning large vertical gaps on each page for missing anchors.
    Uses span-level parsing near left margin to find additional question numbers."""
    try:
        doc = fitz.open(pdf_path)
        # Group by page
        by_page: Dict[int, List[Tuple[int, float]]] = {}
        for p, q, y in anchors:
            by_page.setdefault(p, []).append((q, y))

        refined: List[Tuple[int, int, float]] = anchors[:]
        for p in range(len(doc)):
            page = doc.load_page(p)
            page_h = page.rect.height
            page_w = page.rect.width
            lst = sorted(by_page.get(p, []), key=lambda t: t[1])
            # Build ranges to scan: from top to first, between anchors, last to bottom
            scan_ranges: List[Tuple[float, float]] = []
            if not lst:
                scan_ranges.append((page_h * 0.05, page_h * 0.95))
            else:
                scan_ranges.append((max(0.05 * page_h, lst[0][1] - 40.0), lst[0][1] - 6.0))
                for i in range(len(lst) - 1):
                    y1 = lst[i][1]
                    y2 = lst[i + 1][1]
                    if (y2 - y1) > (0.16 * page_h):  # large gap -> likely missed anchors
                        scan_ranges.append((y1 + 8.0, y2 - 8.0))
                scan_ranges.append((lst[-1][1] + 6.0, page_h * 0.95))

            pdict = page.get_text("dict")
            for (y_start, y_end) in scan_ranges:
                if y_end <= y_start:
                    continue
                for block in pdict.get("blocks", []):
                    for line in block.get("lines", []):
                        if not line.get("spans"):
                            continue
                        bbox = line.get("bbox", [0, 0, 0, 0])
                        ly0 = float(bbox[1]); lx0 = float(bbox[0])
                        if ly0 < y_start or ly0 > y_end:
                            continue
                        # left margin guard
                        if lx0 > (page_w * 0.18):
                            continue
                        # combine first two spans for robustness
                        t0 = (line["spans"][0].get("text") or "").strip()
                        t1 = (line["spans"][1].get("text") or "").strip() if len(line["spans"]) > 1 else ""
                        tcat = (t0 + t1).strip()
                        m = re.match(r"^(\d{1,3})\s*[\)\.]", tcat)
                        if not m:
                            continue
                        qn = int(m.group(1))
                        if qn <= 0 or qn > 400:
                            continue
                        refined.append((p, qn, ly0))

        # Dedup and sort refined anchors
        by_key: Dict[Tuple[int, int], float] = {}
        for p, q, y in refined:
            k = (p, q)
            if k not in by_key or y + DEDUP_EPS_PTS < by_key[k]:
                by_key[k] = y
        out = [(k[0], k[1], y) for k, y in by_key.items()]
        out.sort(key=lambda t: (t[0], t[2]))
        return out
    except Exception:
        logger.debug("[Format2] refine_anchors failed", exc_info=True)
        return anchors

def apply_answers_to_db(answers: Dict[int, str], course=None, subject: Optional[str] = None) -> Dict[str, any]:
    """
    Apply answer mapping to existing Question records in database.
    
    Args:
        answers: Dictionary mapping question numbers to correct answers
        course: Course instance to filter questions
        
    Returns:
        Dictionary with update counts and missing question numbers
    """
    try:
        # Import Question inside the function to avoid circular imports
        from exam.models import Question
        
        logger.info(f"Applying {len(answers)} answers to database")
        updated_count = 0
        missing_questions = []
        
        with transaction.atomic():
            for q_num, correct_option in answers.items():
                try:
                    # Find question by number and course
                    filter_kwargs = {'question_number': q_num}
                    if course:
                        filter_kwargs['course'] = course
                    if subject:
                        filter_kwargs['subject'] = subject
                    
                    question = Question.objects.get(**filter_kwargs)
                    question.correct_answer = correct_option
                    question.save()
                    updated_count += 1
                    logger.debug(f"Updated question {q_num} with answer {correct_option}")
                    
                except Question.DoesNotExist:
                    missing_questions.append(q_num)
                    logger.warning(f"Question {q_num} not found in database")
        
        logger.info(f"Answer application completed: {updated_count} updated, {len(missing_questions)} missing")
        return {
            'updated': updated_count,
            'missing': missing_questions
        }
        
    except Exception as e:
        logger.exception(f"Failed to apply answers to database: {e}")
        return {'updated': 0, 'missing': []}


def process_exam_pdfs(
    questions_pdf_path: str,
    answers_pdf_path: str = None,
    expected_questions: int = None,
    course=None,
    dpi: int = 300,
    overwrite: bool = False,
    subject: Optional[str] = None,
) -> Dict[str, any]:
    """
    Main orchestrator function for the complete PDF processing pipeline.
    
    Args:
        questions_pdf_path: Path to questions PDF
        answers_pdf_path: Path to answers PDF (optional)
        expected_questions: Expected number of questions for validation
        dpi: Resolution for image conversion (default 300)
        overwrite: Whether to overwrite existing good crops
        
    Returns:
        Dictionary with processing results and comprehensive pipeline report
    """
    # Initialize all report variables to avoid NameError
    errors = []
    warnings = []
    skipped_questions = []
    expanded_questions = []
    missing_qnums = []
    last_question_options_end_crop = None
    debug_images = []
    
    # Type check for expected_questions parameter
    if not isinstance(expected_questions, int) or expected_questions <= 0:
        error_msg = f"expected_questions must be a positive integer, got: {expected_questions} (type: {type(expected_questions).__name__})"
        logger.error(error_msg)
        errors.append(error_msg)
        return {
            'success': False,
            'error': error_msg,
            'pipeline_report': {
                'errors': errors,
                'warnings': warnings,
                'created_images': 0,
                'db_created': 0,
                'db_updated': 0,
                'db_skipped': 0,
                'skipped_questions': skipped_questions,
                'expanded_questions': expanded_questions,
                'missing_qnums': missing_qnums,
                'last_question_options_end_crop': last_question_options_end_crop,
                'debug_images': debug_images
            }
        }
    
    try:
        logger.info(f"Starting complete PDF processing pipeline")
        logger.info(f"Questions PDF: {questions_pdf_path}")
        logger.info(f"Answers PDF: {answers_pdf_path}")
        logger.info(f"Expected questions: {expected_questions}")
        logger.info(f"DPI: {dpi}")
        logger.info(f"Overwrite existing: {overwrite}")
        if subject:
            logger.info(f"Subject mode: {subject}")
        
        # Create staging directory
        staging_dir = create_staging_directory()
        logger.info(f"Staging directory: {staging_dir}")
        
        # Step 1: Render PDF pages to images with dynamic scaling
        logger.info("Step 1: Rendering PDF pages to images with dynamic scaling")
        image_paths, pil_images, scale_factors = render_pdf_pages(questions_pdf_path, dpi)
        if not image_paths:
            error_msg = "Failed to render PDF pages to images"
            errors.append(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'pipeline_report': {
                    'errors': errors,
                    'warnings': warnings,
                    'created_images': 0,
                    'db_created': 0,
                    'db_updated': 0,
                    'db_skipped': 0,
                    'skipped_questions': skipped_questions,
                    'expanded_questions': expanded_questions,
                    'missing_qnums': missing_qnums,
                    'last_question_options_end_crop': last_question_options_end_crop,
                    'debug_images': debug_images
                }
            }
        
        # Step 2: Extract question anchors using PyMuPDF with line-aware filtering
        logger.info("Step 2: Extracting question anchors with PyMuPDF line-aware filtering")
        anchors = extract_number_positions_from_pdf(questions_pdf_path)
        
        # Step 3: Fallback to OCR if insufficient anchors
        if len(anchors) < expected_questions * 0.8:  # Less than 80% of expected
            logger.warning(f"PyMuPDF found only {len(anchors)} anchors, falling back to OCR")
            warnings.append(f"PyMuPDF found only {len(anchors)} anchors, using OCR fallback")
            
            # Use OCR fallback for each page
            ocr_anchors = []
            for page_idx, image in enumerate(pil_images):
                ocr_positions = ocr_find_numbers_on_image(image)
                for q_num, x_px, y_px in ocr_positions:
                    # Convert pixel y to PDF points using scale factor
                    if page_idx < len(scale_factors):
                        scale_x, scale_y = scale_factors[page_idx]
                        y_pt = y_px / scale_y
                        ocr_anchors.append((page_idx, q_num, y_pt))
            
            # Merge and deduplicate anchors
            all_anchors = anchors + ocr_anchors
            seen = set()
            merged_anchors = []
            for anchor in all_anchors:
                key = (anchor[0], anchor[1])  # (page, question_num)
                if key not in seen:
                    merged_anchors.append(anchor)
                    seen.add(key)
            
            anchors = merged_anchors
            logger.info(f"Combined anchors: {len(anchors)} total")
        
        if not anchors:
            error_msg = "No question anchors found with PyMuPDF or OCR fallback"
            errors.append(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'pipeline_report': {
                    'errors': errors,
                    'warnings': warnings,
                    'created_images': 0,
                    'db_created': 0,
                    'db_updated': 0,
                    'db_skipped': 0,
                    'skipped_questions': skipped_questions,
                    'expanded_questions': expanded_questions,
                    'missing_qnums': missing_qnums,
                    'last_question_options_end_crop': last_question_options_end_crop,
                    'debug_images': debug_images
                }
            }
        
        # Step 4: Validate question count
        distinct_questions = len(set(anchor[1] for anchor in anchors))
        if distinct_questions != expected_questions:
            warning_msg = f"Found {distinct_questions} questions, expected {expected_questions}"
            warnings.append(warning_msg)
            logger.warning(warning_msg)
        
        # Step 5: Crop questions with look-ahead algorithm
        logger.info("Step 5: Cropping questions with look-ahead algorithm")
        crops, skipped_pairs = crop_questions_from_images(
            questions_pdf_path, image_paths, pil_images, anchors, staging_dir, overwrite, validate_mode='format1', subject=subject
        )
        if not crops:
            error_msg = "Failed to crop question images"
            errors.append(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'pipeline_report': {
                    'errors': errors,
                    'warnings': warnings,
                    'created_images': 0,
                    'db_created': 0,
                    'db_updated': 0,
                    'db_skipped': 0,
                    'skipped_questions': skipped_questions,
                    'expanded_questions': expanded_questions,
                    'missing_qnums': missing_qnums,
                    'last_question_options_end_crop': last_question_options_end_crop,
                    'debug_images': debug_images
                }
            }
        
        # Step 6: Save crops to database atomically
        logger.info("Step 6: Saving crops to database")
        db_result = save_question_images_to_db(crops, course=course, overwrite=overwrite, subject=subject)
        
        # Step 7: Parse and apply answers if provided
        answer_result = {'updated': 0, 'missing': []}
        if answers_pdf_path:
            logger.info("Step 7: Parsing and applying answers")
            answers = parse_answers_pdf(answers_pdf_path)
            if answers:
                answer_result = apply_answers_to_db(answers, course=course, subject=subject)
                missing_qnums = answer_result.get('missing', [])
            else:
                warnings.append("No answers found in answers PDF")
        
        # Step 8: Collect debug images
        debug_images = []
        for i in range(1, 10):  # Assume max 9 pages
            debug_path = os.path.join(staging_dir, f'page_{i}_debug.jpg')
            if os.path.exists(debug_path):
                debug_images.append(f'page_{i}_debug.jpg')
        
        # Step 9: Generate comprehensive pipeline report
        logger.info("Step 9: Generating pipeline report")
        pipeline_report = {
            'timestamp': datetime.now().isoformat(),
            'created_images': len(crops),
            'crops_generated': len(crops),
            'db_created': db_result.get('created', 0),
            'db_updated': db_result.get('updated', 0),
            'db_skipped': db_result.get('skipped', 0),
            'skipped_questions': [q for q, _ in skipped_pairs],
            'skipped_with_reasons': skipped_pairs,
            'expanded_questions': expanded_questions,
            'missing_qnums': missing_qnums,
            'last_question_options_end_crop': last_question_options_end_crop,
            'debug_images': debug_images,
            'errors': errors,
            'warnings': warnings,
            'staging_directory': staging_dir,
            'anchors_found': len(anchors),
            'answer_updates': answer_result.get('updated', 0),
            'subject': subject
        }
        
        # Save pipeline report to JSON file
        report_path = os.path.join(staging_dir, 'pipeline_report.json')
        with open(report_path, 'w') as f:
            json.dump(pipeline_report, f, indent=2)
        # include the report path for admin UI convenience
        pipeline_report['report_path'] = report_path
        
        logger.info(f"Complete pipeline completed successfully: {len(crops)} questions processed")
        
        return {
            'success': True,
            'questions_created': db_result.get('created', 0),
            'questions_updated': db_result.get('updated', 0),
            'answers_updated': answer_result.get('updated', 0),
            'skipped_questions': [q for q, _ in skipped_pairs],
            'skipped_with_reasons': skipped_pairs,
            'pipeline_report': pipeline_report
        }
        
    except Exception as e:
        error_msg = f"Pipeline failed: {e}"
        logger.exception(error_msg)
        errors.append(error_msg)
        return {
            'success': False,
            'error': error_msg,
            'pipeline_report': {
                'errors': errors,
                'warnings': warnings,
                'created_images': 0,
                'db_created': 0,
                'db_updated': 0,
                'db_skipped': 0,
                'skipped_questions': skipped_questions,
                'expanded_questions': expanded_questions,
                'missing_qnums': missing_qnums,
                'last_question_options_end_crop': last_question_options_end_crop,
                'debug_images': debug_images
            }
        }


def extract_number_positions_from_pdf_format2(pdf_path: str) -> List[Tuple[int, int, float]]:
    """
    Extract question number positions from PDF using Format 2 parsing rules.
    This is a placeholder implementation that mirrors Format 1 logic.
    Customize this function based on the specific layout and parsing requirements of Format 2.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of tuples: (page_index, question_number, y_point)
    """
    try:
        doc = fitz.open(pdf_path)
        positions = []
        
        logger.info(f"Extracting question anchors from Format 2 PDF using PyMuPDF: {pdf_path}")
        
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            page_height = page.rect.height
            page_width = page.rect.width
            words = page.get_text('words')  # Returns (x0, y0, x1, y1, word, block_no, line_no, word_no)
            
            # Group words into lines by y-coordinate with tolerance
            line_tolerance = 5.0  # Points
            lines = {}
            
            for word_data in words:
                x0, y0, x1, y1, word = word_data[:5]
                
                # Find the closest line for this word
                line_y = None
                for existing_y in lines.keys():
                    if abs(existing_y - y0) <= line_tolerance:
                        line_y = existing_y
                        break
                
                if line_y is None:
                    line_y = y0
                
                if line_y not in lines:
                    lines[line_y] = []
                lines[line_y].append((x0, y0, x1, y1, word))
            
            # Process each line for question numbers
            # TODO: Customize this section for Format 2 specific parsing rules
            for line_y, line_words in lines.items():
                # Sort words in line by x-coordinate
                line_words.sort(key=lambda w: w[0])
                
                # Check if first word in line matches question number pattern
                # NOTE: This uses Format 1 logic - modify for Format 2 requirements
                if line_words:
                    first_word = line_words[0]
                    x0, y0, x1, y1, word = first_word
                    
                    # Match question numbers with regex: ^\s*(\d{1,3})\s*[)\.]
                    # TODO: Update regex pattern for Format 2 if different
                    match = re.match(r'^\s*(\d{1,3})\s*[)\.]', word.strip())
                    if match:
                        question_num = int(match.group(1))
                        
                        # Apply Format 2 specific margin filters
                        # TODO: Adjust these margins based on Format 2 layout
                        # Accept only tokens within 22% of the left margin
                        if x0 > (page_width * 0.22):
                            continue
                        
                        # Exclude anything in the top 3% or bottom 5% of page height
                        if y0 < (page_height * 0.03) or y0 > (page_height * 0.95):
                            continue
                        
                        # Skip tokens within the rightmost 12% (typical page numbers)
                        if x0 > (page_width * 0.88):
                            continue
                        
                        # Skip any line containing a slash / (avoid "1/2", "2/2")
                        line_text = ' '.join([w[4] for w in line_words])
                        if '/' in line_text:
                            continue
                        
                        # Skip lines starting with option markers like (a), (b), (c), (d)
                        if any(re.match(r'^\s*\([a-dA-D]\)', w[4].strip()) for w in line_words):
                            continue
                        
                        positions.append((page_index, question_num, y0))
                        logger.debug(f"Found Format 2 question {question_num} at page {page_index+1}, y={y0}")
        
        # For each (page, qnum), keep the topmost y and drop near-duplicates
        best_by_key: Dict[Tuple[int, int], float] = {}
        for page_idx, q_num, y_coord in positions:
            key = (page_idx, q_num)
            if key not in best_by_key:
                best_by_key[key] = y_coord
            else:
                # keep the smallest y (topmost)
                if y_coord + DEDUP_EPS_PTS < best_by_key[key]:
                    best_by_key[key] = y_coord
        deduplicated = [(k[0], k[1], y) for k, y in best_by_key.items()]
        deduplicated.sort(key=lambda x: (x[0], x[2]))
        
        logger.info(f"Found {len(deduplicated)} Format 2 question anchors using PyMuPDF (after deduplication)")
        return deduplicated
        
    except Exception as e:
        logger.exception(f"Failed to extract Format 2 anchors with PyMuPDF: {e}")
        return []


def parse_answers_pdf_format2(answers_pdf_path: str) -> Dict[int, str]:
    """
    Parse answers from Format 2 PDF using pure text extraction.
    Handles the specific format: 26.(c), 27.(C), 28.(c), 29.(a), 30.(b), 31.(b)
    
    Args:
        answers_pdf_path: Path to the answers PDF file
        
    Returns:
        Dictionary mapping question numbers to correct answers (A, B, C, D)
    """
    try:
        logger.info(f"Parsing Format 2 answers from PDF: {answers_pdf_path}")
        doc = fitz.open(answers_pdf_path)
        answers = {}
        
        # Extract all text from PDF with clean processing
        full_text = ""
        for page in doc:
            page_text = page.get_text("text")
            full_text += page_text + "\n"
        
        logger.debug(f"Extracted text length: {len(full_text)} characters")
        
        # Clean up the text - remove extra whitespace and normalize
        full_text = re.sub(r'\s+', ' ', full_text.strip())
        
        # Primary pattern for Format 2: number.(letter) format
        # Matches: 26.(c), 27.(C), 28.(c), etc.
        primary_pattern = r'(\d{1,3})\.\(([a-dA-D])\)'
        
        # Secondary patterns for variations
        secondary_patterns = [
            r'(\d{1,3})\s*\.\s*\(([a-dA-D])\)',  # With spaces: 26. (c)
            r'(\d{1,3})\.([a-dA-D])(?!\w)',       # Without parentheses: 26.c
            r'(\d{1,3})\s*\)\s*([a-dA-D])',       # With closing paren: 26) c
            r'(\d{1,3})\s*[\.-]\s*([a-dA-D])(?!\w)', # General separator: 26-c, 26.c
        ]
        
        # Try primary pattern first
        matches = re.findall(primary_pattern, full_text, re.IGNORECASE)
        logger.debug(f"Primary pattern found {len(matches)} matches")
        
        for match in matches:
            q_num = int(match[0])
            option = match[1].upper()
            if option in ['A', 'B', 'C', 'D']:
                answers[q_num] = option
                logger.debug(f"Format 2 answer: Question {q_num} = {option}")
        
        # If primary pattern didn't find enough answers, try secondary patterns
        if len(answers) < 5:  # Assume we should find at least 5 answers
            logger.info("Primary pattern found few matches, trying secondary patterns")
            
            for pattern in secondary_patterns:
                matches = re.findall(pattern, full_text, re.IGNORECASE)
                logger.debug(f"Secondary pattern '{pattern}' found {len(matches)} matches")
                
                for match in matches:
                    q_num = int(match[0])
                    option = match[1].upper()
                    if option in ['A', 'B', 'C', 'D'] and q_num not in answers:
                        answers[q_num] = option
                        logger.debug(f"Format 2 answer (secondary): Question {q_num} = {option}")
        
        # Additional validation: check for line-by-line parsing
        if len(answers) < 3:  # If still very few matches, try line-by-line
            logger.info("Few matches found, attempting line-by-line parsing")
            lines = full_text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Try to match the entire line against our patterns
                for pattern in [primary_pattern] + secondary_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        q_num = int(match.group(1))
                        option = match.group(2).upper()
                        if option in ['A', 'B', 'C', 'D'] and q_num not in answers:
                            answers[q_num] = option
                            logger.debug(f"Format 2 answer (line-by-line): Question {q_num} = {option}")
                        break
        
        doc.close()
        
        # Sort answers by question number for logging
        sorted_answers = dict(sorted(answers.items()))
        logger.info(f"Successfully parsed {len(sorted_answers)} Format 2 answers")
        
        if sorted_answers:
            logger.info(f"Answer range: Q{min(sorted_answers.keys())} to Q{max(sorted_answers.keys())}")
            # Log first few answers for verification
            sample_answers = list(sorted_answers.items())[:5]
            logger.debug(f"Sample answers: {sample_answers}")
        
        return sorted_answers
        
    except Exception as e:
        logger.exception(f"Failed to parse Format 2 answers PDF: {e}")
        return {}


def process_exam_pdfs_format2(
    questions_pdf_path: str,
    answers_pdf_path: str = None,
    expected_questions: int = None,
    course=None,
    dpi: int = 300,
    overwrite: bool = False,
    subject: Optional[str] = None,
) -> Dict[str, any]:
    """
    Main orchestrator function for the Format 2 PDF processing pipeline.
    This mirrors the structure of process_exam_pdfs() but uses Format 2 specific parsing functions.
    
    Args:
        questions_pdf_path: Path to questions PDF
        answers_pdf_path: Path to answers PDF (optional)
        expected_questions: Expected number of questions for validation
        dpi: Resolution for image conversion (default 300)
        overwrite: Whether to overwrite existing good crops
        subject: Subject classification for questions
        
    Returns:
        Dictionary with processing results and comprehensive pipeline report
    """
    # Initialize all report variables to avoid NameError
    errors = []
    warnings = []
    skipped_questions = []
    expanded_questions = []
    missing_qnums = []
    last_question_options_end_crop = None
    debug_images = []
    
    # Type check for expected_questions parameter
    if not isinstance(expected_questions, int) or expected_questions <= 0:
        error_msg = f"expected_questions must be a positive integer, got: {expected_questions} (type: {type(expected_questions).__name__})"
        logger.error(error_msg)
        errors.append(error_msg)
        return {
            'success': False,
            'error': error_msg,
            'pipeline_report': {
                'errors': errors,
                'warnings': warnings,
                'created_images': 0,
                'db_created': 0,
                'db_updated': 0,
                'db_skipped': 0,
                'skipped_questions': skipped_questions,
                'expanded_questions': expanded_questions,
                'missing_qnums': missing_qnums,
                'last_question_options_end_crop': last_question_options_end_crop,
                'debug_images': debug_images
            }
        }
    
    try:
        logger.info(f"Starting Format 2 PDF processing pipeline")
        logger.info(f"Questions PDF: {questions_pdf_path}")
        logger.info(f"Answers PDF: {answers_pdf_path}")
        logger.info(f"Expected questions: {expected_questions}")
        logger.info(f"DPI: {dpi}")
        logger.info(f"Overwrite existing: {overwrite}")
        if subject:
            logger.info(f"Subject mode: {subject}")
        
        # Create staging directory
        staging_dir = create_staging_directory()
        logger.info(f"Format 2 staging directory: {staging_dir}")
        
        # Step 1: Render PDF pages to images with dynamic scaling (reuse existing function)
        logger.info("Step 1: Rendering Format 2 PDF pages to images with dynamic scaling")
        image_paths, pil_images, scale_factors = render_pdf_pages(questions_pdf_path, dpi)
        if not image_paths:
            error_msg = "Failed to render Format 2 PDF pages to images"
            errors.append(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'pipeline_report': {
                    'errors': errors,
                    'warnings': warnings,
                    'created_images': 0,
                    'db_created': 0,
                    'db_updated': 0,
                    'db_skipped': 0,
                    'skipped_questions': skipped_questions,
                    'expanded_questions': expanded_questions,
                    'missing_qnums': missing_qnums,
                    'last_question_options_end_crop': last_question_options_end_crop,
                    'debug_images': debug_images
                }
            }
        
        # Step 2: Extract question anchors using Format 2 specific parsing
        logger.info("Step 2: Extracting Format 2 question anchors with specialized parsing")
        anchors = extract_number_positions_from_pdf_format2(questions_pdf_path)
        # Refinement pass: fill large gaps with potential missed anchors
        anchors = refine_anchors_format2(questions_pdf_path, anchors)
        
        # Step 3: Fallback to OCR if insufficient anchors (reuse existing OCR function)
        if len(anchors) < expected_questions * 0.8:  # Less than 80% of expected
            logger.warning(f"Format 2 PyMuPDF found only {len(anchors)} anchors, falling back to OCR")
            warnings.append(f"Format 2 PyMuPDF found only {len(anchors)} anchors, using OCR fallback")
            
            # Use OCR fallback for each page (reuse existing function)
            ocr_anchors = []
            for page_idx, image in enumerate(pil_images):
                ocr_positions = ocr_find_numbers_on_image(image)
                for q_num, x_px, y_px in ocr_positions:
                    # Convert pixel y to PDF points using scale factor
                    if page_idx < len(scale_factors):
                        scale_x, scale_y = scale_factors[page_idx]
                        y_pt = y_px / scale_y
                        ocr_anchors.append((page_idx, q_num, y_pt))
            
            # Merge and deduplicate anchors
            all_anchors = anchors + ocr_anchors
            seen = set()
            merged_anchors = []
            for anchor in all_anchors:
                key = (anchor[0], anchor[1])  # (page, question_num)
                if key not in seen:
                    merged_anchors.append(anchor)
                    seen.add(key)
            
            anchors = merged_anchors
            logger.info(f"Combined Format 2 anchors: {len(anchors)} total")
        
        if not anchors:
            error_msg = "No Format 2 question anchors found with PyMuPDF or OCR fallback"
            errors.append(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'pipeline_report': {
                    'errors': errors,
                    'warnings': warnings,
                    'created_images': 0,
                    'db_created': 0,
                    'db_updated': 0,
                    'db_skipped': 0,
                    'skipped_questions': skipped_questions,
                    'expanded_questions': expanded_questions,
                    'missing_qnums': missing_qnums,
                    'last_question_options_end_crop': last_question_options_end_crop,
                    'debug_images': debug_images
                }
            }
        
        # Step 4: Validate question count
        distinct_questions = len(set(anchor[1] for anchor in anchors))
        if distinct_questions != expected_questions:
            warning_msg = f"Format 2: Found {distinct_questions} questions, expected {expected_questions}"
            warnings.append(warning_msg)
            logger.warning(warning_msg)
        
        # Step 5: Crop questions with look-ahead algorithm (reuse existing function)
        logger.info("Step 5: Cropping Format 2 questions with look-ahead algorithm")
        crops, skipped_pairs = crop_questions_from_images(
            questions_pdf_path, image_paths, pil_images, anchors, staging_dir, overwrite, validate_mode='format2', subject=subject
        )
        if not crops:
            error_msg = "Failed to crop Format 2 question images"
            errors.append(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'pipeline_report': {
                    'errors': errors,
                    'warnings': warnings,
                    'created_images': 0,
                    'db_created': 0,
                    'db_updated': 0,
                    'db_skipped': 0,
                    'skipped_questions': skipped_questions,
                    'expanded_questions': expanded_questions,
                    'missing_qnums': missing_qnums,
                    'last_question_options_end_crop': last_question_options_end_crop,
                    'debug_images': debug_images
                }
            }
        
        # Step 6: Save crops to database atomically (reuse existing function)
        logger.info("Step 6: Saving Format 2 crops to database")
        db_result = save_question_images_to_db(crops, course=course, overwrite=overwrite, subject=subject)
        
        # Step 7: Parse and apply answers if provided using Format 2 specific parsing
        answer_result = {'updated': 0, 'missing': []}
        if answers_pdf_path:
            logger.info("Step 7: Parsing and applying Format 2 answers")
            answers = parse_answers_pdf_format2(answers_pdf_path)
            if answers:
                answer_result = apply_answers_to_db(answers, course=course, subject=subject)
                missing_qnums = answer_result.get('missing', [])
            else:
                warnings.append("No answers found in Format 2 answers PDF")
        
        # Step 8: Collect debug images (reuse existing logic)
        debug_images = []
        for i in range(1, 10):  # Assume max 9 pages
            debug_path = os.path.join(staging_dir, f'page_{i}_debug.jpg')
            if os.path.exists(debug_path):
                debug_images.append(f'page_{i}_debug.jpg')
        
        # Step 9: Generate comprehensive pipeline report
        logger.info("Step 9: Generating Format 2 pipeline report")
        pipeline_report = {
            'timestamp': datetime.now().isoformat(),
            'format': 'Format 2',
            'created_images': len(crops),
            'crops_generated': len(crops),
            'db_created': db_result.get('created', 0),
            'db_updated': db_result.get('updated', 0),
            'db_skipped': db_result.get('skipped', 0),
            'skipped_questions': [q for q, _ in skipped_pairs],
            'skipped_with_reasons': skipped_pairs,
            'expanded_questions': expanded_questions,
            'missing_qnums': missing_qnums,
            'last_question_options_end_crop': last_question_options_end_crop,
            'debug_images': debug_images,
            'errors': errors,
            'warnings': warnings,
            'staging_directory': staging_dir,
            'anchors_found': len(anchors),
            'answer_updates': answer_result.get('updated', 0),
            'subject': subject
        }
        
        # Save pipeline report to JSON file
        report_path = os.path.join(staging_dir, 'pipeline_report_format2.json')
        with open(report_path, 'w') as f:
            json.dump(pipeline_report, f, indent=2)
        # include the report path for admin UI convenience
        pipeline_report['report_path'] = report_path
        
        logger.info(f"Format 2 pipeline completed successfully: {len(crops)} questions processed")
        
        return {
            'success': True,
            'questions_created': db_result.get('created', 0),
            'questions_updated': db_result.get('updated', 0),
            'answers_updated': answer_result.get('updated', 0),
            'skipped_questions': [q for q, _ in skipped_pairs],
            'skipped_with_reasons': skipped_pairs,
            'pipeline_report': pipeline_report
        }
        
    except Exception as e:
        error_msg = f"Format 2 pipeline failed: {e}"
        logger.exception(error_msg)
        errors.append(error_msg)
        return {
            'success': False,
            'error': error_msg,
            'pipeline_report': {
                'errors': errors,
                'warnings': warnings,
                'created_images': 0,
                'db_created': 0,
                'db_updated': 0,
                'db_skipped': 0,
                'skipped_questions': skipped_questions,
                'expanded_questions': expanded_questions,
                'missing_qnums': missing_qnums,
                'last_question_options_end_crop': last_question_options_end_crop,
                'debug_images': debug_images
            }
        }


# Re-export for backward compatibility
__all__ = ['process_exam_pdfs', 'process_exam_pdfs_format2']
