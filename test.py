import open_clip


model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
