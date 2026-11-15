import torch
from gliner import GLiNER


def load_gliner_model(model_name: str = "gliner-community/gliner_small-v2.5", device: str | None = None) -> GLiNER:
    if device is None or (device == "cuda" and not torch.cuda.is_available()):
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GLiNER.from_pretrained(model_name)  # loads the GLiNER model
    return model.to(device)


def predict_entities(model: GLiNER, text: str, labels: list[str], threshold: float = 0.5) -> list[dict]:
    return model.predict_entities(text, labels, threshold=threshold)  # predicts only a text


def predict_dataset(model: GLiNER, dataset: list[dict], labels: list[str], threshold: float = 0.5) -> list[dict]:
    results = []

    for item in dataset:
        text = item["text"]
        target_labels = labels if labels is not None else item.get("labels", [])

        entities = predict_entities(model, text, target_labels, threshold=threshold)

        results.append({
            "id": item["id"],
            "text": text,
            "entities": entities
        })

    return results
