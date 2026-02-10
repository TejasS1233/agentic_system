"""
QR Code Generator Tool - Encode text, URLs, or data into QR code images.

Uses the qrcode library with Pillow for rendering.
"""

import os
import json
from typing import Optional
from pydantic import BaseModel, Field


class QRCodeGeneratorToolArgs(BaseModel):
    data: str = Field(
        ...,
        description="Text, URL, or data to encode in the QR code.",
    )
    size: Optional[int] = Field(
        10,
        description="Box size in pixels per module (default 10, range 1-40).",
    )
    border: Optional[int] = Field(
        4,
        description="Border width in modules (default 4).",
    )
    fill_color: Optional[str] = Field(
        "black",
        description="QR code foreground color (default 'black').",
    )
    back_color: Optional[str] = Field(
        "white",
        description="QR code background color (default 'white').",
    )
    output_path: Optional[str] = Field(
        None,
        description="Output file path for the PNG image.",
    )


class QRCodeGeneratorTool:
    """
    Generate QR code images from text, URLs, or any data.

    Features:
    - Customizable size, border, and colors
    - Auto error correction level
    - Saves as PNG
    - Supports URLs, plain text, vCard, WiFi config, etc.
    """

    name = "qr_code_generator"
    description = (
        "Generate a QR code image from text, URL, or data. "
        "Input any text like a URL 'https://example.com' or plain text "
        "and get a PNG QR code image. Customizable colors and size."
    )
    args_schema = QRCodeGeneratorToolArgs

    def __init__(self, output_dir: str = "/output"):
        self.output_dir = output_dir
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError:
            self.output_dir = "/tmp"

    def run(
        self,
        data: str,
        size: int = 10,
        border: int = 4,
        fill_color: str = "black",
        back_color: str = "white",
        output_path: str = None,
    ) -> str:
        """Generate a QR code from input data.

        Args:
            data: Text/URL to encode
            size: Module pixel size (1-40)
            border: Border width in modules
            fill_color: Foreground color
            back_color: Background color
            output_path: Output PNG path

        Returns:
            JSON with success status and output path.
        """
        if not data or not data.strip():
            return json.dumps({
                "success": False,
                "error": "No data provided to encode.",
            }, indent=2)

        try:
            import qrcode
            from qrcode.constants import ERROR_CORRECT_H
        except ImportError:
            return json.dumps({
                "success": False,
                "error": "qrcode library not available. Install with: pip install qrcode[pil]",
            }, indent=2)

        size = max(1, min(40, size or 10))
        border = max(0, min(10, border or 4))

        try:
            qr = qrcode.QRCode(
                version=None,  # Auto-determine
                error_correction=ERROR_CORRECT_H,
                box_size=size,
                border=border,
            )
            qr.add_data(data)
            qr.make(fit=True)

            img = qr.make_image(fill_color=fill_color, back_color=back_color)

            # Save — always under /output/ so it persists on host volume
            if output_path and not output_path.startswith(self.output_dir):
                output_path = os.path.join(self.output_dir, os.path.basename(output_path))
            if not output_path:
                existing = [f for f in os.listdir(self.output_dir) if f.startswith("qrcode_") and f.endswith(".png")]
                idx = len(existing) + 1
                output_path = os.path.join(self.output_dir, f"qrcode_{idx}.png")

            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

            img.save(output_path)
            file_size = os.path.getsize(output_path)

            return json.dumps({
                "success": True,
                "data_encoded": data[:100] + ("..." if len(data) > 100 else ""),
                "data_length": len(data),
                "output_path": output_path,
                "image_size": f"{img.size[0]}x{img.size[1]}",
                "file_size_bytes": file_size,
                "message": f"QR code generated and saved to {output_path}",
            }, indent=2)

        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e),
            }, indent=2)
