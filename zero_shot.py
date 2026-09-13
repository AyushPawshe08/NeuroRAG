"""
Zero-shot classification using BioMedCLIP's text encoder.

This is NOT the retrieval system (that's Qdrant + ingest_qdrant.py). This is a
cheap, instant sanity-check baseline: compare the uploaded scan's image
embedding directly against a handful of text-label embeddings, with no
database lookup at all.

Kept separate from embedder.py on purpose:
    - embedder.py  -> owns the loaded model, knows HOW to turn things into vectors
    - zero_shot.py -> owns the DOMAIN logic: which labels/prompts we care about,
                       and how to turn similarity scores into a ranked guess

This way, embedder.py stays reusable and dumb, and any changes to label
wording or the class list happen in exactly one place.
"""
import torch

from embedder import BioMedCLIPEmbedder

# Map each dataset class to a natural-language prompt. Wording matters somewhat
# for CLIP-style models — short, descriptive phrases tend to work better than
# a single bare word.
CLASS_PROMPTS = {
    "glioma": "a brain MRI scan showing a glioma tumor",
    "meningioma": "a brain MRI scan showing a meningioma tumor",
    "pituitary": "a brain MRI scan showing a pituitary tumor",
    "notumor": "a normal brain MRI scan with no tumor",
}


class ZeroShotClassifier:
    _instance = None

    def __init__(self):
        # Reuse the already-loaded BioMedCLIP model/tokenizer instead of
        # loading the weights a second time.
        self.embedder = BioMedCLIPEmbedder.get()
        self.labels = list(CLASS_PROMPTS.keys())
        prompts = list(CLASS_PROMPTS.values())
        # Text embeddings only need to be computed once, ever - the prompts
        # never change - so we cache them at startup rather than on every request.
        self.text_embeddings = torch.tensor(self.embedder.embed_texts(prompts))

    @classmethod
    def get(cls) -> "ZeroShotClassifier":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def classify(self, image) -> dict:
        """
        Takes a PIL image, returns a ranked list of {label, score} plus the
        single best guess. Scores are cosine similarities (already normalized
        embeddings), roughly in the 0-1 range for this kind of model.
        """
        image_embedding = torch.tensor(self.embedder.embed_image(image))

        # Cosine similarity between one image vector and every text vector.
        # Both are already L2-normalized, so a plain dot product IS the
        # cosine similarity - no extra division needed.
        similarities = self.text_embeddings @ image_embedding

        ranked = sorted(
            zip(self.labels, similarities.tolist()),
            key=lambda pair: pair[1],
            reverse=True,
        )

        return {
            "best_guess": ranked[0][0],
            "scores": [{"label": label, "score": round(score, 4)} for label, score in ranked],
        }


if __name__ == "__main__":
    # Quick smoke test: run `python zero_shot.py path/to/some_scan.jpg`
    import sys
    from PIL import Image

    if len(sys.argv) < 2:
        print("Usage: python zero_shot.py <image_path>")
        sys.exit(1)

    classifier = ZeroShotClassifier.get()
    result = classifier.classify(Image.open(sys.argv[1]))
    print(result)