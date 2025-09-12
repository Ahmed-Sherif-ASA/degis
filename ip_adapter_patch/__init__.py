"""
DEGIS IP-Adapter Patch Module
=============================

This module contains DEGIS enhancements for IP-Adapter via monkey patching.
"""

# Auto-apply IP-Adapter patches when this module is imported
from .ip_adapter_monkey_patch import apply_patches

# Apply patches immediately
apply_patches()
