"""
Model management utilities for handling external checkpoints and caching.
"""
import os
import torch
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    """Centralized model management for external checkpoints and caching."""
    
    def __init__(self, 
                 cache_dir: Optional[str] = None,
                 models_dir: Optional[str] = None):
        """
        Initialize model manager with caching and storage directories.
        
        Args:
            cache_dir: Directory for HuggingFace cache (defaults to ~/.cache/huggingface)
            models_dir: Directory for custom model checkpoints (defaults to ./models)
        """
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface")
        self.models_dir = models_dir or "./models"
        
        # Ensure directories exist
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
    
    def get_model_path(self, model_name: str, model_type: str = "checkpoint") -> str:
        """
        Get the full path for a model file.
        
        Args:
            model_name: Name of the model (e.g., "ip-adapter_sd15", "color_head_best")
            model_type: Type of model ("checkpoint", "hf_model", "custom")
            
        Returns:
            Full path to the model file
        """
        if model_type == "checkpoint":
            return os.path.join(self.models_dir, f"{model_name}.pth")
        elif model_type == "bin":
            return os.path.join(self.models_dir, f"{model_name}.bin")
        elif model_type == "hf_model":
            # HuggingFace models are handled by the library
            return model_name
        else:
            return os.path.join(self.models_dir, f"{model_name}.{model_type}")
    
    def download_hf_model(self, model_id: str, force_download: bool = False) -> str:
        """
        Download a HuggingFace model and return its local path.
        
        Args:
            model_id: HuggingFace model identifier
            force_download: Whether to force re-download
            
        Returns:
            Local path to the downloaded model
        """
        try:
            from huggingface_hub import snapshot_download
            local_path = snapshot_download(
                repo_id=model_id,
                cache_dir=self.cache_dir,
                force_download=force_download
            )
            logger.info(f"Downloaded {model_id} to {local_path}")
            return local_path
        except ImportError:
            logger.warning("huggingface_hub not available, using default caching")
            return model_id
    
    def save_checkpoint(self, 
                       model: torch.nn.Module, 
                       model_name: str,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save a PyTorch model checkpoint.
        
        Args:
            model: PyTorch model to save
            model_name: Name for the saved model
            metadata: Optional metadata to save alongside the model
            
        Returns:
            Path to the saved checkpoint
        """
        checkpoint_path = self.get_model_path(model_name, "checkpoint")
        
        # Prepare checkpoint data
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'metadata': metadata or {}
        }
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        return checkpoint_path
    
    def load_checkpoint(self, 
                       model_name: str, 
                       device: Optional[torch.device] = None) -> Dict[str, Any]:
        """
        Load a PyTorch model checkpoint.
        
        Args:
            model_name: Name of the model to load
            device: Device to load the model on
            
        Returns:
            Dictionary containing model state dict and metadata
        """
        checkpoint_path = self.get_model_path(model_name, "checkpoint")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        device = device or torch.device("cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
        return checkpoint
    
    def list_available_models(self) -> Dict[str, list]:
        """
        List all available models in the models directory.
        
        Returns:
            Dictionary categorizing available models by type
        """
        models = {
            "checkpoints": [],
            "bin_files": [],
            "other": []
        }
        
        if not os.path.exists(self.models_dir):
            return models
        
        for file in os.listdir(self.models_dir):
            file_path = os.path.join(self.models_dir, file)
            if os.path.isfile(file_path):
                if file.endswith('.pth'):
                    models["checkpoints"].append(file)
                elif file.endswith('.bin'):
                    models["bin_files"].append(file)
                else:
                    models["other"].append(file)
        
        return models

# Global model manager instance
_model_manager = None

def get_model_manager(cache_dir: Optional[str] = None, 
                     models_dir: Optional[str] = None) -> ModelManager:
    """Get or create the global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager(cache_dir=cache_dir, models_dir=models_dir)
    return _model_manager

def setup_model_environment(cache_dir: Optional[str] = None,
                           models_dir: Optional[str] = None):
    """Set up the model environment with proper caching."""
    manager = get_model_manager(cache_dir=cache_dir, models_dir=models_dir)
    return manager
