"""
DEGIS IP-Adapter Patch Module
=============================

This module contains DEGIS enhancements for IP-Adapter via patching.
"""

# Auto-apply DEGIS IP-Adapter patches when this module is imported
from .degis_ip_adapter_patch import apply_patches

# Apply patches immediately
apply_patches()
