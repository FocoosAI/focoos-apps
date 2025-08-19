"""
Base application class for Focoos AI applications.

This module provides the foundational BaseApp class that handles model loading
and initialization for all Focoos applications. It supports loading models
from Focoos Hub with various runtime options.
"""

from typing import Any, Optional
from focoos.hub import FocoosHUB
from focoos.model_manager import ModelManager
from focoos.models.focoos_model import FocoosModel
from focoos.ports import RuntimeType


class BaseApp:
    """
    Base class for all Focoos AI applications.
    
    This class provides common functionality for loading and managing Focoos models from the Focoos ecosystem.
    It supports loading models from Focoos Hub with various runtime options.
    
    Attributes:
        model: The FocoosModel instance.
    """

    def __init__(
        self,
        model_name: str = None,
        api_key: Optional[str] = None,
        runtime: Optional[str] = "cpu",
        image_size: Optional[int] = None,
    ):
        """
        Initialize the BaseApp.
        
        Args:
            model_name: Model name, path, or hub reference (e.g., "hub://username/model_ref")
            api_key: The API key for the Focoos Hub.
            runtime: The runtime type to use for the model (cpu, cuda, tensorrt)
            image_size: The image size to use for the model.
        """
        
        self.model: FocoosModel | None = None

        try:
            if api_key is not None:
                # Initialize HUB only if an API key is provided
                hub = FocoosHUB(api_key=api_key)
                self.model = ModelManager.get(model_name, hub=hub)
            else:
                    self.model = ModelManager.get(model_name)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
        
        # Map runtime string to RuntimeType
        if runtime == "cpu":
            runtime_type = RuntimeType.ONNX_CPU
        elif runtime == "cuda":
            runtime_type = RuntimeType.ONNX_CUDA32
        elif runtime == "tensorrt":
            runtime_type = RuntimeType.ONNX_TRT16
        else:
            raise ValueError(f"Invalid runtime: {runtime}, must be one of: cpu, cuda, tensorrt")
        
        self.model = self.model.export(runtime_type=runtime_type, image_size=image_size)


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


