"""
Image Processing Tool - Comprehensive image manipulation utility.
"""

import json
import ast
from pathlib import Path
from typing import Optional, Dict, Union
from pydantic import BaseModel, Field


class ImageProcessorToolArgs(BaseModel):
    image_path: str = Field(..., description="Path to the input image file")
    operation: str = Field(
        ...,
        description="Operation: 'resize', 'crop', 'rotate', 'filter', 'grayscale', 'convert', 'adjust'",
    )
    params: Optional[str] = Field(
        None,
        description="JSON string or dictionary of parameters for the operation (e.g. {'width': 100, 'height': 100} for resize)",
    )
    output_path: Optional[str] = Field(None, description="Path for output image")


class ImageProcessorTool:
    name = "process_image"
    description = "Advanced image processing: resize, crop, rotate, flip, filter (blur, sharpen, etc.), adjust (brightness, contrast), convert format, and grayscale."
    args_schema = ImageProcessorToolArgs

    def run(
        self,
        image_path: str,
        operation: str,
        params: Union[str, Dict] = None,
        output_path: str = None,
    ) -> str:
        """Process an image with the specified operation and parameters."""
        try:
            from PIL import Image, ImageFilter, ImageEnhance, ImageOps
        except ImportError:
            return (
                "Error: Pillow library not installed. Install with: pip install Pillow"
            )

        # Validate input path
        input_path = Path(image_path)
        if not input_path.exists():
            return f"Error: Image file not found: {image_path}"

        # Parse parameters
        parameters = {}
        if params:
            if isinstance(params, dict):
                parameters = params
            elif isinstance(params, str):
                try:
                    # Try JSON first
                    parameters = json.loads(params)
                except json.JSONDecodeError:
                    try:
                        # Try literal eval for python dict string
                        parameters = ast.literal_eval(params)
                    except (ValueError, SyntaxError):
                        return f"Error: Invalid params format. Must be JSON or dict string. Got: {params}"

        try:
            # Open image
            img = Image.open(input_path)
            original_format = img.format

            op_lower = operation.lower()

            # --- Operations ---

            if op_lower in [
                "grayscale",
                "greyscale",
                "bw",
                "black_and_white",
                "blackandwhite",
                "monochrome",
            ] or ("black" in op_lower and "white" in op_lower):
                processed = img.convert("L")
                operation = "grayscale"

            elif op_lower == "resize":
                width = parameters.get("width")
                height = parameters.get("height")
                scale = parameters.get("scale")

                if scale:
                    width = int(img.width * float(scale))
                    height = int(img.height * float(scale))
                elif not width and not height:
                    # Default: 50% scale
                    width = int(img.width * 0.5)
                    height = int(img.height * 0.5)

                if not width:
                    width = int(img.width * (height / img.height))
                if not height:
                    height = int(img.height * (width / img.width))

                processed = img.resize((width, height), Image.Resampling.LANCZOS)

            elif op_lower == "crop":
                # params: left, top, right, bottom OR box=(left, top, right, bottom)
                box = parameters.get("box")
                if not box:
                    left = parameters.get("left", 0)
                    top = parameters.get("top", 0)
                    right = parameters.get("right", img.width)
                    bottom = parameters.get("bottom", img.height)
                    box = (left, top, right, bottom)

                processed = img.crop(box)

            elif op_lower == "rotate":
                angle = parameters.get("angle", 90)
                expand = parameters.get("expand", True)
                processed = img.rotate(angle, expand=expand)

            elif op_lower == "flip":
                direction = parameters.get("direction", "horizontal")
                if "hor" in direction:
                    processed = ImageOps.mirror(img)
                elif "ver" in direction:
                    processed = ImageOps.flip(img)
                else:
                    return "Error: Flip direction must be 'horizontal' or 'vertical'"

            elif op_lower == "filter":
                filter_type = parameters.get("type", "blur").upper()
                filter_map = {
                    "BLUR": ImageFilter.BLUR,
                    "CONTOUR": ImageFilter.CONTOUR,
                    "DETAIL": ImageFilter.DETAIL,
                    "EDGE_ENHANCE": ImageFilter.EDGE_ENHANCE,
                    "EDGE_ENHANCE_MORE": ImageFilter.EDGE_ENHANCE_MORE,
                    "EMBOSS": ImageFilter.EMBOSS,
                    "FIND_EDGES": ImageFilter.FIND_EDGES,
                    "SHARPEN": ImageFilter.SHARPEN,
                    "SMOOTH": ImageFilter.SMOOTH,
                    "SMOOTH_MORE": ImageFilter.SMOOTH_MORE,
                }

                if filter_type not in filter_map:
                    return f"Error: Unknown filter type '{filter_type}'. Supported: {', '.join(filter_map.keys())}"

                processed = img.filter(filter_map[filter_type])

            elif op_lower == "adjust":
                brightness = parameters.get("brightness")
                contrast = parameters.get("contrast")
                sharpness = parameters.get("sharpness")
                color = parameters.get("color")

                processed = img
                if brightness is not None:
                    processed = ImageEnhance.Brightness(processed).enhance(
                        float(brightness)
                    )
                if contrast is not None:
                    processed = ImageEnhance.Contrast(processed).enhance(
                        float(contrast)
                    )
                if sharpness is not None:
                    processed = ImageEnhance.Sharpness(processed).enhance(
                        float(sharpness)
                    )
                if color is not None:
                    processed = ImageEnhance.Color(processed).enhance(float(color))

            elif op_lower == "convert":
                # Just conversion (handled by save)
                processed = img
            else:
                return f"Error: Unknown operation '{operation}'. Supported: grayscale, resize, crop, rotate, flip, filter, adjust, convert"

            # --- Suffix Logic ---
            if op_lower == "convert":
                new_ext = parameters.get("format", "png").lower()
                suffix = f".{new_ext}"
            else:
                suffix = input_path.suffix

            # --- Output Path ---
            if not output_path:
                # Check for standard container output directory
                container_output = Path("/output")
                if container_output.exists() and container_output.is_dir():
                    output_path = str(
                        container_output / f"{input_path.stem}_{operation}{suffix}"
                    )
                else:
                    # Fallback to same directory (local execution)
                    output_path = str(
                        input_path.with_stem(f"{input_path.stem}_{operation}")
                    )

            # --- Save ---
            # Ensure output directory exists (if local)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            processed.save(output_path)
            return f"Success: Image processed with '{operation}' and saved to: {output_path}"

        except Exception as e:
            return f"Error processing image: {str(e)}"


if __name__ == "__main__":
    print("ImageProcessorTool ready.")
