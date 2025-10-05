"""
DEGIS IP-Adapter Implementation
==============================

This module provides a complete IP-Adapter implementation with DEGIS enhancements.
No patching needed - all classes are available directly.

Usage:
    from ip_adapter_patch import IPAdapter, IPAdapterXL
    or
    from ip_adapter_patch.degis_ip_adapter_patch import IPAdapter, IPAdapterXL
"""

from .degis_ip_adapter_patch import (
    ImageProjModel,
    MLPProjModel, 
    EmbeddingAdapter,
    IPAdapter,
    IPAdapterXL,
    IPAdapterPlus,
    IPAdapterFull,
    IPAdapterPlusXL
)

__all__ = [
    "ImageProjModel",
    "MLPProjModel",
    "EmbeddingAdapter", 
    "IPAdapter",
    "IPAdapterXL",
    "IPAdapterPlus",
    "IPAdapterFull",
    "IPAdapterPlusXL"
]
