"""
FastAPI backend.

Exposes:
    POST /predict   -> upload an MRI image, get back the top-K most similar
                        historical cases from Qdrant (label + score + metadata)

Run with:
    uvicorn main:app --reload
"""
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from PIL import Image
from qdrant_client import QdrantClient

from config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION, TOP_K, DATASET_DIR
from embedder import BioMedCLIPEmbedder
from zero_shot import ZeroShotClassifier
from gemini_report import generate_report

app = FastAPI(title="Brain Tumor RAG API")

# Serves the dataset images over HTTP at /images/<label>/<filename>,
# so the Gradio dashboard (running on a different machine once deployed)
# can actually load and display them.
app.mount("/images", StaticFiles(directory=DATASET_DIR), name="images")

# These are created once, when the server starts — not on every request.
# Loading BioMedCLIP or reconnecting to Qdrant per-request would be very slow.
embedder = BioMedCLIPEmbedder.get()
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
zero_shot_classifier = ZeroShotClassifier.get()  # reuses embedder's model, no reload


@app.get("/health")
def health():
    """Simple check that the server is alive and the model loaded correctly."""
    return {"status": "ok", "collection": QDRANT_COLLECTION}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an uploaded MRI image, embeds it with BioMedCLIP, and returns the
    top-K most visually similar historical cases stored in Qdrant.
    """
    # 1. Read the uploaded bytes and open them as a PIL image
    contents = await file.read()
    try:
        image = Image.open(BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    # 2. Turn the image into a 512-dim vector, same way we did during ingestion
    query_vector = embedder.embed_image(image)

    # 3. Ask Qdrant for the most similar vectors it has stored
    #    (newer qdrant-client versions renamed .search() to .query_points())
    response = qdrant_client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=TOP_K,
    )
    results = response.points

    # 4. Shape the response into something simple and JSON-friendly
    matches = [
        {
            "label": hit.payload.get("label"),
            "filename": hit.payload.get("filename"),
            "image_url": f"/images/{hit.payload.get('relative_path')}",
            "similarity_score": round(hit.score, 4),
        }
        for hit in results
    ]

    # 5. Primary diagnosis: weighted by AVERAGE similarity per label, not raw
    #    count and not sum. Average matters because summing would let several
    #    decent-but-not-great matches (e.g. three at 0.96) outweigh a single
    #    near-perfect match (e.g. one at 1.0) - averaging asks "how similar
    #    are matches of this label, typically?" instead.
    label_totals: dict[str, float] = {}
    label_counts: dict[str, int] = {}
    for m in matches:
        label_totals[m["label"]] = label_totals.get(m["label"], 0.0) + m["similarity_score"]
        label_counts[m["label"]] = label_counts.get(m["label"], 0) + 1
    label_averages = {label: label_totals[label] / label_counts[label] for label in label_totals}
    primary_diagnosis = max(label_averages, key=label_averages.get) if label_averages else None

    # 6. Independent cross-check: zero-shot text classification, no Qdrant
    #    involved at all. If this disagrees with the retrieval result, that's
    #    a useful signal that the case is ambiguous or borderline.
    zero_shot_result = zero_shot_classifier.classify(image)

    # 7. Generative reasoning: Gemini writes an explanation grounded in the
    #    evidence we already retrieved (steps 3-6), not a fresh guess.
    try:
        gemini_explanation = generate_report(image, primary_diagnosis, matches, zero_shot_result)
    except Exception as e:
        gemini_explanation = f"Gemini report unavailable: {e}"

    return {
        "primary_diagnosis": primary_diagnosis,
        "top_matches": matches,
        "zero_shot_check": zero_shot_result,
        "gemini_explanation": gemini_explanation,
    }