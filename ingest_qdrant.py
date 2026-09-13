"""
Walks the dataset folder structure:

    DATASET_DIR/
        glioma/
            img1.jpg
            img2.jpg
        meningioma/
            ...
        notumor/
            ...
        pituitary/
            ...

Embeds every image with BioMedCLIP and upserts it into a Qdrant collection,
storing the class label, filename, and original path as payload metadata.
This becomes the "historical case" evidence bank the RAG system retrieves from.

Run:
    python ingest_qdrant.py
"""
import os
import uuid
from pathlib import Path

from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from tqdm import tqdm

from config import (
    DATASET_DIR,
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    EMBED_DIM,
)
from embedder import BioMedCLIPEmbedder

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def get_class_folders(dataset_dir: str) -> list[Path]:
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"DATASET_DIR '{dataset_dir}' does not exist. "
            f"Set it in your .env file to point at your dataset root."
        )
    return [p for p in root.iterdir() if p.is_dir()]


def ensure_collection(client: QdrantClient):
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION in existing:
        print(f"[ingest] Collection '{QDRANT_COLLECTION}' already exists — will upsert into it.")
        return
    print(f"[ingest] Creating collection '{QDRANT_COLLECTION}' (dim={EMBED_DIM})")
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=qmodels.VectorParams(
            size=EMBED_DIM,
            distance=qmodels.Distance.COSINE,
        ),
    )


def main():
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    ensure_collection(client)

    embedder = BioMedCLIPEmbedder.get()
    class_folders = get_class_folders(DATASET_DIR)

    if not class_folders:
        print(f"[ingest] No class subfolders found under {DATASET_DIR}. Nothing to do.")
        return

    print(f"[ingest] Found classes: {[f.name for f in class_folders]}")

    batch_points = []
    BATCH_SIZE = 64
    total_indexed = 0

    for class_folder in class_folders:
        label = class_folder.name
        image_paths = [
            p for p in class_folder.iterdir()
            if p.suffix.lower() in VALID_EXTS
        ]
        print(f"[ingest] {label}: {len(image_paths)} images")

        for img_path in tqdm(image_paths, desc=f"Embedding {label}"):
            try:
                image = Image.open(img_path)
                vector = embedder.embed_image(image)
            except Exception as e:
                print(f"  [skip] {img_path} -> {e}")
                continue

            point = qmodels.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "label": label,
                    "filename": img_path.name,
                    "path": str(img_path.resolve()),
                },
            )
            batch_points.append(point)

            if len(batch_points) >= BATCH_SIZE:
                client.upsert(collection_name=QDRANT_COLLECTION, points=batch_points)
                total_indexed += len(batch_points)
                batch_points = []

    if batch_points:
        client.upsert(collection_name=QDRANT_COLLECTION, points=batch_points)
        total_indexed += len(batch_points)

    print(f"[ingest] Done. Indexed {total_indexed} images into '{QDRANT_COLLECTION}'.")


if __name__ == "__main__":
    main()