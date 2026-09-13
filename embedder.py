"""
Wraps BioMedCLIP to turn an MRI image into a normalized 512-dim feature vector.

BioMedCLIP is a CLIP-style model pretrained on biomedical image-text pairs,
so it captures medically-relevant visual features far better than a generic
ImageNet-pretrained CLIP would.
"""
import torch
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer

from config import BIOMEDCLIP_MODEL


class BioMedCLIPEmbedder:
    _instance = None  # simple singleton so the model loads only once per process

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[BioMedCLIPEmbedder] Loading {BIOMEDCLIP_MODEL} on {self.device} ...")
        self.model, self.preprocess = create_model_from_pretrained(BIOMEDCLIP_MODEL)
        self.tokenizer = get_tokenizer(BIOMEDCLIP_MODEL)
        self.model.to(self.device).eval()
        print("[BioMedCLIPEmbedder] Model ready.")

    @classmethod
    def get(cls) -> "BioMedCLIPEmbedder":
        """Reuse a single loaded model instead of reloading weights every call."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @torch.no_grad()
    def embed_image(self, image: Image.Image) -> list[float]:
        """
        Takes a PIL image, returns an L2-normalized embedding as a plain
        python list (ready to hand to Qdrant).
        """
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
        """
        Takes a list of text prompts (e.g. ["glioma MRI", "normal brain MRI"])
        and returns one L2-normalized embedding per prompt, using the SAME
        loaded model as embed_image — so text and image vectors land in the
        same comparable space, and we never load BioMedCLIP twice.
        """
        text_tokens = self.tokenizer(texts).to(self.device)
        features = self.model.encode_text(text_tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().tolist()


if __name__ == "__main__":
    # Quick smoke test: run `python embedder.py path/to/some_scan.jpg`
    import sys

    if len(sys.argv) < 2:
        print("Usage: python embedder.py <image_path>")
        sys.exit(1)

    embedder = BioMedCLIPEmbedder.get()
    vec = embedder.embed_image_path(sys.argv[1])
    print(f"Embedding length: {len(vec)}")
    print(f"First 5 values: {vec[:5]}")