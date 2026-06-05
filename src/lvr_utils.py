import math
from typing import List, Tuple, Union
import numpy as np


def expand_bbox(bbox, expansion_factor=1.3):
    """Expand a normalized [x1, y1, x2, y2] bbox around its center."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1
    new_w, new_h = w * expansion_factor, h * expansion_factor
    return [
        max(0.0, cx - new_w / 2),
        max(0.0, cy - new_h / 2),
        min(1.0, cx + new_w / 2),
        min(1.0, cy + new_h / 2),
    ]

class QwenVLBboxTokenMapper:
    """
    Maps bounding box coordinates to visual token indices for QWEN 2.5 VL models.
    
    QWEN 2.5 VL uses a vision transformer that processes images in patches,
    with each patch corresponding to one or more visual tokens.
    """
    
    def __init__(self, 
                 patch_size: int = 14,
                 spatial_merge_size: int = 2):
        """
        Initialize the mapper with QWEN 2.5 VL vision tower parameters.
        Image dimensions will be set dynamically when processing each image.
        
        Args:
            patch_size: Size of each patch in pixels
            spatial_merge_size: Spatial merge factor (typically 2 for QWEN 2.5)
        """
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        
        # These will be set dynamically for each image
        self.image_height = None
        self.image_width = None
        self.grid_height = None
        self.grid_width = None
        self.token_grid_height = None
        self.token_grid_width = None
        self.total_tokens = None
        
    def _setup_image_dimensions(self, image_height: int, image_width: int):
        """
        Set up grid dimensions for a specific image size.
        
        Args:
            image_height: Height of the input image in pixels
            image_width: Width of the input image in pixels
        """
        self.image_height = image_height
        self.image_width = image_width
        
        # Calculate grid dimensions after patching
        self.grid_height = self.image_height // self.patch_size
        self.grid_width = self.image_width // self.patch_size
        
        # Calculate final token grid after spatial merging
        self.token_grid_height = self.grid_height // self.spatial_merge_size
        self.token_grid_width = self.grid_width // self.spatial_merge_size
        
        self.total_tokens = self.token_grid_height * self.token_grid_width
        
    def bbox_to_token_indices(self, 
                            bbox: List[float], 
                            image_height: int,
                            image_width: int,
                            bbox_format: str = "xyxy",
                            return_grid_coords: bool = False) -> Union[List[int], Tuple[List[int], List[Tuple[int, int]]]]:
        """
        Convert bounding box coordinates to visual token indices.
        
        Args:
            bbox: Bounding box coordinates [a, b, c, d]
            image_height: Height of the input image in pixels
            image_width: Width of the input image in pixels
            bbox_format: Format of bbox - "xyxy" (x1,y1,x2,y2) or "xywh" (x,y,w,h)
            return_grid_coords: If True, also return grid coordinates
            
        Returns:
            List of token indices, optionally with grid coordinates
        """
        # Setup dimensions for this specific image
        self._setup_image_dimensions(image_height, image_width)
        
        # Normalize and convert bbox format
        if bbox_format == "xyxy":
            x1, y1, x2, y2 = bbox
        elif bbox_format == "xywh":
            x, y, w, h = bbox
            x1, y1, x2, y2 = x, y, x + w, y + h
        else:
            raise ValueError("bbox_format must be 'xyxy' or 'xywh'")
        
        # NOTE: bounding boxes are expected to be normalized to [0,1]. Coordinates are
        # not modified here; the caller must pre-normalize. Conversion to token indices
        # happens after the images are processed.
        if max(x1, y1, x2, y2) > 1.0:
            x1 /= image_width
            y1 /= image_height
            x2 /= image_width
            y2 /= image_height
        
        # Clamp coordinates to valid range
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(1, x2), min(1, y2)
        
        # Convert to token grid coordinates
        # Map from image coordinates to token grid coordinates
        token_x1 = int(x1 * self.token_grid_width)
        token_y1 = int(y1 * self.token_grid_height)
        token_x2 = min(int(math.ceil(x2 * self.token_grid_width)), self.token_grid_width)
        token_y2 = min(int(math.ceil(y2 * self.token_grid_height)), self.token_grid_height)
        
        # Ensure we have at least one token
        if token_x2 <= token_x1:
            token_x2 = token_x1 + 1
        if token_y2 <= token_y1:
            token_y2 = token_y1 + 1
        
        # Generate token indices and grid coordinates
        token_indices = []
        grid_coords = []
        
        for y in range(token_y1, token_y2):
            for x in range(token_x1, token_x2):
                # Convert 2D grid position to 1D token index
                token_idx = y * self.token_grid_width + x
                token_indices.append(token_idx)
                grid_coords.append((y, x))
        
        if return_grid_coords:
            return token_indices, grid_coords
        return token_indices
    
    def token_index_to_bbox(self, token_indices: List[int]) -> List[float]:
        """
        Convert token indices back to bounding box.
        
        Args:
            token_indices: List of token indices
            
        Returns:
            Bounding box in normalized xyxy format [x1, y1, x2, y2]
        """
        if not token_indices:
            return [0, 0, 0, 0]
        
        # Convert token indices to grid coordinates
        grid_coords = []
        for idx in token_indices:
            y = idx // self.token_grid_width
            x = idx % self.token_grid_width
            grid_coords.append((y, x))
        
        # Find bounding box of grid coordinates
        min_y = min(coord[0] for coord in grid_coords)
        max_y = max(coord[0] for coord in grid_coords)
        min_x = min(coord[1] for coord in grid_coords)
        max_x = max(coord[1] for coord in grid_coords)
        
        # Convert back to normalized image coordinates
        x1 = min_x / self.token_grid_width
        y1 = min_y / self.token_grid_height
        x2 = (max_x + 1) / self.token_grid_width
        y2 = (max_y + 1) / self.token_grid_height
        
        return [x1, y1, x2, y2]
