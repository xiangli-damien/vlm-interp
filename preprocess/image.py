# preprocess/image.py
"""
Image utilities
"""

from io import BytesIO
from pathlib import Path
from typing import Tuple, Union, Optional
from PIL import Image, UnidentifiedImageError
import requests


__all__ = ["load_image"]


def _open_from_url(url: str, timeout: int = 20) -> Image.Image:
    resp = requests.get(url, stream=True, timeout=timeout)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content))


def _open_from_path(path: Union[str, Path]) -> Image.Image:
    return Image.open(path)


def load_image(
    source: Union[str, Path, BytesIO, Image.Image],
    resize_to: Optional[Tuple[int, int]] = None,
    convert_mode: str = "RGB"
) -> Image.Image:
    try:
        if isinstance(source, Image.Image):
            img = source.copy()

        elif isinstance(source, (str, Path)) and str(source).startswith(("http://", "https://")):
            img = _open_from_url(str(source))

        elif isinstance(source, (str, Path)):
            img = _open_from_path(Path(source))

        elif isinstance(source, BytesIO):
            source.seek(0)
            img = Image.open(source)

        else:
            raise TypeError(f"Unsupported image source: {type(source)!r}")

        # Ensure valid image content
        if img is None:
            raise ValueError("Image loading returned None unexpectedly.")

        # Convert mode if needed
        if convert_mode and img.mode != convert_mode:
            img = img.convert(convert_mode)

        # Resize if needed
        if resize_to:
            img = img.resize(resize_to, Image.Resampling.LANCZOS)

        return img

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Image file not found: {source}") from e
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to download image from URL: {e}") from e
    except UnidentifiedImageError:
        raise ValueError("Source does not contain a valid image") from None
    except Exception as e:
        raise ValueError(f"Failed to load or process image: {e}") from e