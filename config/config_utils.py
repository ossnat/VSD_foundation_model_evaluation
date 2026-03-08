
def get_embedding_dim(config):
  # Determine embedding_dim based on backbone name for model initialization
    backbone_name = config["model"]["backbone"]
    if 'dino' in backbone_name:
        return 768
    elif 'resnet' in backbone_name or 'frodo' in backbone_name:
        return 512
    else:
      print(f"Warning: Unknown backbone '{backbone_name}', using default embedding_dim=768")
      return config["model"].get("embedding_dim", 768)
