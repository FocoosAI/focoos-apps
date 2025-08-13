"""
Base application class for Focoos AI applications.

This module provides the foundational BaseApp class that handles model loading
and initialization for all Focoos applications. It supports loading models
from Focoos Hub references or local paths with various runtime options.
"""

from typing import Any, Optional, List, Union
from focoos.hub import FocoosHUB
from focoos.model_manager import ModelManager
from focoos.ports import RuntimeType


def map_runtime_string_to_type(runtime_str: Optional[str]) -> Optional[RuntimeType]:
    """
    Map runtime string options to RuntimeType enum values.
    
    Args:
        runtime_str: String option ('cpu', 'cuda', 'tensorrt') or None
        
    Returns:
        RuntimeType enum value or None
    """
    if runtime_str is None:
        return None
    
    runtime_mapping = {
        'cpu': RuntimeType.ONNX_CPU,
        'cuda': RuntimeType.ONNX_CUDA32,
        'tensorrt': RuntimeType.ONNX_TRT16,
    }
    
    return runtime_mapping.get(runtime_str.lower())


class BaseApp:
    """
    Base class for all Focoos AI applications.
    
    This class provides common functionality for loading and managing AI models
    from the Focoos ecosystem. It supports both local model paths and remote
    model references from Focoos Hub.
    
    Attributes:
        model: The loaded AI model instance
        cls_names: List of class names for the model's output classes
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        api_key: Optional[str] = None,
        model_ref: Optional[str] = None,
        runtime: Optional[Union[str, RuntimeType]] = None,
        image_size: Optional[int] = None,
    ):
        self.model: Any | None = None
        self.cls_names: List[str] = []
        
        # Map runtime string to RuntimeType if needed
        runtime_type = None
        if isinstance(runtime, str):
            runtime_type = map_runtime_string_to_type(runtime)
        else:
            runtime_type = runtime
        
        # Prefer model_ref + runtime if provided; fallback to raw path loader
        if model_ref:
            self.load_model_from_ref(
                model_ref=model_ref,
                api_key=api_key,
                runtime_type=runtime_type,
                image_size=image_size,
            )
        elif model_path:
            # TODO: implement model loading from path
            raise NotImplementedError("Model loading from path is not implemented")


    def load_model_from_ref(
        self,
        model_ref: str,
        api_key: Optional[str] = None,
        runtime_type: RuntimeType = RuntimeType.ONNX_TRT16,
        image_size: Optional[int] = None,
    ) -> bool:
        """
        Initialize a model via Focoos SDK using `ModelManager.get(model_ref)`.
        Export to a specific runtime (`RuntimeType`) with `image_size`.
        Returns True on success.
        """
        try:
            if api_key is not None:
                # Initialize HUB only if an API key is provided
                hub = FocoosHUB(api_key=api_key)
                model = ModelManager.get(model_ref, hub=hub)
            else:
                model = ModelManager.get(model_ref)

            # Export to a specific runtime if requested (e.g., "cpu", "cuda", "tensorrt")
            if runtime_type:
                model = model.export(runtime_type=runtime_type, image_size=image_size)
            self.model = model
            self.cls_names = model.model_info.classes
        except Exception as e:
            print(f"Error loading model from ref: {e}")
            return False

        return True

    def process(self, *args: Any, **kwargs: Any):
        """
        Process method should be implemented by each App subclass.
        
        This is a placeholder method that should be overridden by subclasses
        to implement their specific processing logic.
        
        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        raise NotImplementedError("Subclasses must implement the process method")


