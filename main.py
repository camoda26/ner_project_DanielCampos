from gliner import GLiNER

model = GLiNER.from_pretrained("gliner-community/gliner_small-v2.5")
text = "El Consorci d'Educació de Barcelona adjudica a Joan Garcia el contrato 2025-AB el 14 de marzo de 2025."
labels = ["organization", "person", "date", "contract_id"]
entities = model.predict_entities(text, labels, threshold=0.5)

for ent in entities:
    print(ent)
