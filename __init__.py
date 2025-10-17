# __init__.py
# Package initializer for the WAN 2.2 Resolution Helper (16x) ComfyUI node.

from .wan_resolution_helper import (
    NODE_CLASS_MAPPINGS as _NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _NODE_DISPLAY_NAME_MAPPINGS,
)

# Expose the maps for ComfyUI's node loader
NODE_CLASS_MAPPINGS = dict(_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS = dict(_NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
