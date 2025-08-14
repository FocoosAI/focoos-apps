"""
Base application class for Focoos AI applications.

This module provides the foundational BaseApp class that handles model loading
and initialization for all Focoos applications. It supports loading models
from Focoos Hub with various runtime options.
"""

from typing import Any, Optional, List
from focoos.hub import FocoosHUB
from focoos.model_manager import ModelManager
from focoos.ports import RuntimeType


class BaseApp:
    """
    Base class for all Focoos AI applications.
    
    This class provides common functionality for loading and managing AI models
    from the Focoos ecosystem. It supports loading models from Focoos Hub.
    
    Attributes:
        model: The loaded AI model instance
        cls_names: List of class names for the model's output classes
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_ref: Optional[str] = None,
        runtime: Optional[str] = None,
        image_size: Optional[int] = None,
    ):
        self.model: Any | None = None
        self.cls_names: List[str] = []
        
        # Map runtime string to RuntimeType
        if runtime is not None:
            if runtime == "cpu":
                runtime_type = RuntimeType.ONNX_CPU
            elif runtime == "cuda":
                runtime_type = RuntimeType.ONNX_CUDA32
            elif runtime == "tensorrt":
                runtime_type = RuntimeType.ONNX_TRT16
            else:
                raise ValueError(f"Invalid runtime: {runtime}")
        else:
            runtime_type = None
        
        # Load model from Focoos Hub
        self._load_model_from_ref(
            model_ref=model_ref,
            api_key=api_key,
            runtime_type=runtime_type,
            image_size=image_size,
        )


    def _load_model_from_ref(
        self,
        model_ref: str,
        api_key: Optional[str] = None,
        runtime_type: RuntimeType = RuntimeType.ONNX_CPU,
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

            # Export to a specific runtime type if requested
            if runtime_type:
                model = model.export(runtime_type=runtime_type, image_size=image_size)
            self.model = model
            self.cls_names = model.model_info.classes
        except Exception as e:
            print(f"Error loading model from ref: {e}")
            return False

        return True

    def run(self, *args: Any, **kwargs: Any):
        """
        Run method should be implemented by each App subclass.
        
        This is a placeholder method that should be overridden by subclasses
        to implement their specific run logic.
        
        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        raise NotImplementedError("Subclasses must implement the run method")


