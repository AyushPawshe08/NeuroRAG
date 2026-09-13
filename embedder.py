import torch
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer

from config import BIOMEDCLIP_MODEL


class BioMedCLIPEmbedder:
    _instance = None

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[BioMedCLIPEmbedder] Loading {BIOMEDCLIP_MODEL} on {self.device} ...")
        self.model, self.preprocess = create_model_from_pretrained(BIOMEDCLIP_MODEL)
        self.tokenizer = get_tokenizer(BIOMEDCLIP_MODEL)
        self.model.to(self.device).eval()
        print("[BioMedCLIPEmbedder] Model ready.")

    @classmethod
    def get(cls) -> "BioMedCLIPEmbedder":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @torch.no_grad()
    def embed_image(self, image: Image.Image) -> list[float]:
        image = image.convert("RGB")
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        features = self.model.encode_image(img_tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().tolist()

    @torch.no_grad()
    def embed_image_path(self, path: str) -> list[float]:
        return self.embed_image(Image.open(path))

    @torch.no_grad()
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        text_tokens = self.tokenizer(texts).to(self.device)
        features = self.model.encode_text(text_tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().tolist()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python embedder.py <image_path>")
        sys.exit(1)

    embedder = BioMedCLIPEmbedder.get()
    vec = embedder.embed_image_path(sys.argv[1])
    print(f"Embedding length: {len(vec)}")
    print(f"First 5 values: {vec[:5]}")