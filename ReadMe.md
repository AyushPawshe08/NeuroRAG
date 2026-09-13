# 🧠 Brain Tumor RAG — Multimodal Clinical Decision Support

> Retrieval-Augmented Generation for brain MRI analysis. Upload a scan, get back visually similar historical cases *and* an evidence-grounded explanation — not a black-box guess.

![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange)
![Qdrant](https://img.shields.io/badge/vector%20db-Qdrant-dc244c)
![Gemini](https://img.shields.io/badge/LLM-Gemini-8E75B2)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## 🎥 Demo

<!-- Replace this with your actual demo video/GIF -->
**[▶️ Watch the full demo video here](#)**

<!-- Optional: add a GIF preview inline -->
<!-- ![demo](./assets/demo.gif) -->

<!-- Add 1-2 screenshots of the Gradio dashboard here -->
<!-- ![dashboard screenshot](./assets/screenshot.png) -->

---

## 🩺 What problem does this solve?

Standard AI diagnostic tools give a radiologist a label and a confidence score — a black box. This system instead works the way a second opinion actually works: it finds **real, similar historical cases with confirmed diagnoses**, shows them side-by-side with the new scan, and only *then* generates a written explanation grounded in that visible evidence. The radiologist can verify every claim against an actual image, not just trust a number.

## ✨ Features

- 🔍 **Visual similarity search** — BioMedCLIP embeddings + Qdrant vector search retrieve the top-K most similar historical MRI cases
- 🧾 **Evidence-grounded reasoning** — Gemini writes a short diagnostic explanation *based on the retrieved evidence*, explicitly instructed not to invent unsupported claims
- ✅ **Independent cross-check** — a separate zero-shot text-matching pass flags cases where the two methods disagree, surfacing ambiguity instead of hiding it
- ⚖️ **Similarity-weighted voting** — a single near-perfect match correctly outweighs several weaker ones (not a naive majority count)
- 🖼️ **Full visual transparency** — every retrieved comparison case is shown directly in the UI, not just referenced by filename

## 🏗️ Architecture

```
                Upload MRI scan
                       │
                       ▼
              ┌──────────────────┐
              │   Gradio UI       │  (app.py)
              └────────┬──────────┘
                       │
        ┌──────────────┼───────────────┬─────────────────┐
        ▼              ▼               ▼                 ▼
 ┌─────────────┐ ┌───────────────┐ ┌───────────┐  ┌───────────────┐
 │ BioMedCLIP  │ │  Qdrant        │ │ Zero-shot  │  │ Gemini         │
 │ (embedder)  │→│  vector search │ │ cross-check│  │ (reasoning)    │
 └─────────────┘ └───────────────┘ └───────────┘  └───────────────┘
   image → 512-dim   top-K similar    independent      grounded written
   vector             historical      text-based        explanation of
                       cases          sanity check       the evidence
```

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Embeddings | [BioMedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) (via `open_clip`) |
| Vector database | [Qdrant](https://qdrant.tech/) |
| Reasoning | [Gemini](https://ai.google.dev/) (`google-genai`) |
| UI | [Gradio](https://www.gradio.dev/) |
| Backend (local dev) | [FastAPI](https://fastapi.tiangolo.com/) |

## 📁 Dataset structure

Expects the standard 4-class layout:

```
data/train/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/
```

## 🚀 Getting Started

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/brain-mri-rag.git
cd brain-mri-rag
pip install -r requirements.txt
```

### 2. Set up your environment

```bash
cp .env.example .env
```

Edit `.env`:
```dotenv
DATASET_DIR=./data/train
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=brain_mri_cases
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.6-flash
TOP_K=5
```

Get a free Gemini API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

### 3. Start Qdrant

```bash
docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

### 4. Index your dataset

```bash
python ingest_qdrant.py
```

This embeds every image with BioMedCLIP and stores it in Qdrant. Takes ~15 minutes for ~5,600 images on CPU.

### 5. Run it

**Option A — unified app (recommended, matches the deployed version):**
```bash
python app.py
```

**Option B — separate backend + dashboard (for local API testing):**
```bash
# terminal 1
uvicorn main:app --reload
# terminal 2
python gradio_app.py
```

Open the printed local URL (usually `http://127.0.0.1:7860`) and upload a scan.

## 💡 Usage Example

Upload any brain MRI scan through the UI. You'll get back:

```json
{
  "primary_diagnosis": "meningioma",
  "top_matches": [
    {"label": "meningioma", "similarity_score": 1.0},
    {"label": "meningioma", "similarity_score": 0.9779},
    {"label": "pituitary", "similarity_score": 0.96}
  ],
  "zero_shot_check": {"best_guess": "pituitary", "scores": [...]},
  "gemini_explanation": "The retrieval system identified strong matches for meningioma, led by a near-exact match (1.0)..."
}
```

## ⚠️ Known Limitations

This is a decision-support demo, not a validated clinical tool:

- BioMedCLIP is used **off-the-shelf (inference only)** — it was never fine-tuned on this specific dataset, so accuracy on visually ambiguous or out-of-distribution scans is limited
- Trained/indexed on ~1,400 images per class — a small sample compared to production-grade medical AI systems
- Similarity scores drop noticeably on genuinely unseen images (real-world test: ~0.89 vs ~0.98+ on in-distribution images) — this is expected behavior, not a bug, and is exactly why the zero-shot cross-check and Gemini's uncertainty flagging exist
- Not evaluated against a formal held-out test set with reported accuracy metrics (planned improvement)

## ☁️ Deployment

Deployed as a single [Hugging Face Space](https://huggingface.co/spaces) (Gradio SDK) with Qdrant Cloud as the managed vector database — see `app.py` for the self-contained deployment version (no separate backend service required).

## 📄 License

MIT — see [LICENSE](./LICENSE) for details.