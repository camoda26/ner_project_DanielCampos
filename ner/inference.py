from typing import Optional

from gliner import GLiNER


def load_gliner_model(
        model_name: str = "gliner-community/gliner_small-v2.5",
        device: Optional[str] = None
) -> GLiNER:
    model = GLiNER.from_pretrained(model_name)
    if device is not None:
        model = model.to(device)
    return model
