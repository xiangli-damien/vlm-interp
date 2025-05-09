"""
One-stop package for **runtime input preparation**:

    * Image I/O & transforms      →  :pyobj:`preprocess.image`
    * High-level InputBuilder     →  :pyobj:`preprocess.input_builder.prepare_inputs`
    * Vision grid mapping tools   →  :pyobj:`preprocess.mapper.VisionMapper`
"""

from preprocess.image import load_image
from preprocess.input_builder import prepare_inputs
from preprocess.mapper import VisionMapper

__all__ = [
    # low-level
    "load_image",
    # high-level
    "prepare_inputs",
    # mapping tools
    "VisionMapper",
]
