import torch

from embedder import BioMedCLIPEmbedder

CLASS_PROMPTS = {
    "glioma": "a brain MRI scan showing a glioma tumor",
    "meningioma": "a brain MRI scan showing a meningioma tumor",
    "pituitary": "a brain MRI scan showing a pituitary tumor",
    "notumor": "a normal brain MRI scan with no tumor",
}


class ZeroShotClassifier:
    _instance = None

    def __init__(self):
        self.embedder = BioMedCLIPEmbedder.get()
        self.labels = list(CLASS_PROMPTS.keys())
        prompts = list(CLASS_PROMPTS.values())
        self.text_embeddings = torch.tensor(self.embedder.embed_texts(prompts))

    @classmethod
    def get(cls) -> "ZeroShotClassifier":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def classify(self, image) -> dict:
        image_embedding = torch.tensor(self.embedder.embed_image(image))
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
    import sys
    from PIL import Image

    if len(sys.argv) < 2:
        print("Usage: python zero_shot.py <image_path>")
        sys.exit(1)

    classifier = ZeroShotClassifier.get()
    result = classifier.classify(Image.open(sys.argv[1]))
    print(result)