"""
Input Loader - Loads files from the inputs directory and injects them as context.

Supports: .txt, .md, .pdf, .json, .py, .csv, and images (.jpg, .jpeg, .png, .gif, .bmp, .webp)
"""

import os
from pathlib import Path
from typing import Optional, List, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)


class InputLoader:
    """Loads input files from a directory and builds context for the system."""

    TEXT_EXTENSIONS = {".txt", ".md", ".json", ".py", ".csv", ".pdf"}
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | IMAGE_EXTENSIONS

    def __init__(self, input_dir: Path):
        self.input_dir = Path(input_dir)
        self.input_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"InputLoader initialized: {self.input_dir}")

    def has_files(self) -> bool:
        """Check if there are any supported files in the input directory."""
        if not self.input_dir.exists():
            return False
        for f in self.input_dir.iterdir():
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                return True
        return False

    def get_files(self) -> list[Path]:
        """Get list of supported files in the input directory."""
        if not self.input_dir.exists():
            return []
        return [
            f for f in self.input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]

    def get_text_files(self) -> list[Path]:
        """Get list of text-based files (excludes images)."""
        if not self.input_dir.exists():
            return []
        return [
            f for f in self.input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in self.TEXT_EXTENSIONS
        ]

    def get_image_files(self) -> list[Path]:
        """Get list of image files."""
        if not self.input_dir.exists():
            return []
        return [
            f for f in self.input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in self.IMAGE_EXTENSIONS
        ]

    def load_file(self, file_path: Path) -> Optional[str]:
        """Load content from a single file."""
        try:
            suffix = file_path.suffix.lower()

            if suffix in self.IMAGE_EXTENSIONS:
                # For images, return a path reference (tools will process the binary)
                return f"[IMAGE FILE: {file_path.absolute()}]"
            elif suffix == ".pdf":
                return self._load_pdf(file_path)
            else:
                # Text-based files
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()

        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
            return None

    def _load_pdf(self, file_path: Path) -> Optional[str]:
        """Extract text from PDF using PyMuPDF if available."""
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(file_path)
            text = []
            for page in doc:
                text.append(page.get_text())
            doc.close()
            return "\n".join(text)
        except ImportError:
            logger.warning("PyMuPDF not installed. Install with: pip install pymupdf")
            return f"[PDF file: {file_path.name} - install pymupdf to extract text]"
        except Exception as e:
            logger.warning(f"PDF extraction failed: {e}")
            return None

    def build_context(self, max_chars_per_file: int = None, truncate: bool = False) -> Tuple[str, List[Path]]:
        """
        Build context string from all input files.
        
        Args:
            max_chars_per_file: Maximum characters per file (default: no limit)
            truncate: If True, truncate files exceeding max_chars_per_file
        
        Returns:
            Tuple of (context_string, list_of_image_paths)
        """
        files = self.get_files()
        image_files = self.get_image_files()
        
        if not files:
            return "", []

        context_parts = []
        context_parts.append("=" * 60)
        context_parts.append("INPUT DOCUMENTS")
        context_parts.append("=" * 60)
        
        # Add image file info prominently at the top
        if image_files:
            context_parts.append("\n--- IMAGE FILES AVAILABLE ---")
            for img_path in image_files:
                context_parts.append(f"  - {img_path.name}: {img_path.absolute()}")
            context_parts.append("Use these paths with image processing tools.")
            context_parts.append("--- END IMAGE FILES ---\n")

        # Add text file contents
        for file_path in self.get_text_files():
            content = self.load_file(file_path)
            if content:
                # Only truncate if explicitly requested
                if truncate and max_chars_per_file and len(content) > max_chars_per_file:
                    content = content[:max_chars_per_file] + "\n... [TRUNCATED]"

                context_parts.append(f"\n--- FILE: {file_path.name} ---")
                context_parts.append(content)
                context_parts.append(f"--- END: {file_path.name} ---\n")

        context_parts.append("=" * 60)
        context_parts.append("END INPUT DOCUMENTS")
        context_parts.append("=" * 60)

        full_context = "\n".join(context_parts)
        logger.info(f"Built context from {len(files)} file(s) ({len(full_context)} chars, {len(image_files)} images)")
        return full_context, image_files

    def clear_inputs(self):
        """Clear all files from the input directory."""
        files = self.get_files()
        for f in files:
            try:
                f.unlink()
                logger.info(f"Removed: {f.name}")
            except Exception as e:
                logger.warning(f"Failed to remove {f.name}: {e}")
