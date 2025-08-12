from dataclasses import dataclass
from typing import Any, Optional, List
from focoos.hub import FocoosHUB
from focoos.model_manager import ModelManager
from focoos.ports import RuntimeType


class BaseApp:

    def __init__(
        self,
        model_path: Optional[str] = None,
        api_key: Optional[str] = None,
        model_ref: Optional[str] = None,
        runtime_type: Optional[str] = None,
        image_size: Optional[int] = None,
    ):
        self.model: Any | None = None
        self.cls_names: List[str] = []
        
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
        runtime_type: RuntimeType = RuntimeType.TORCHSCRIPT_32,
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

            # Export to a specific runtime if requested (e.g., "TORCHSCRIPT_32", "ONNX_CUDA32", "ONNX_TRT16")
            if runtime_type:
                model = model.export(runtime_type=runtime_type, image_size=image_size)
            self.model = model
            self.cls_names = model.model_info.classes
        except Exception as e:
            print(f"Error loading model from ref: {e}")
            return False

        return True

    def process(self, *args: Any, **kwargs: Any):
        """Process method should be implemented by each App subclass."""


