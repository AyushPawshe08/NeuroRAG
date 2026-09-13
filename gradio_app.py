"""
Gradio dashboard for the Brain Tumor RAG system.

This is a thin UI layer - it does NO AI work itself. It just sends the
uploaded image to the FastAPI backend's /predict endpoint (must already be
running) and displays the JSON response as a readable, visual report.

Run (with `uvicorn main:app --reload` already running in another terminal):
    python gradio_app.py
"""
import io

import gradio as gr
import requests
from PIL import Image

BACKEND_URL = "http://localhost:8000/predict"


def analyze_scan(image: Image.Image):
    """
    Called every time the user clicks "Analyze". Sends the image to FastAPI,
    then reshapes the JSON response into the pieces Gradio needs to display:
    a diagnosis string, a markdown report, a gallery of comparison images,
    and a table of zero-shot scores.
    """
    if image is None:
        return "No image uploaded.", "", [], []

    # Convert the PIL image into raw bytes, the same format a real file
    # upload would send over HTTP.
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    try:
        response = requests.post(
            BACKEND_URL,
            files={"file": ("scan.png", buffer, "image/png")},
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.ConnectionError:
        return (
            "⚠️ Could not reach the backend.",
            "Make sure `uvicorn main:app --reload` is running in another terminal.",
            [],
            [],
        )
    except Exception as e:
        return f"⚠️ Error: {e}", "", [], []

    # --- Primary diagnosis headline ---
    diagnosis_text = f"**Primary diagnosis (retrieval-based):** {result['primary_diagnosis'].upper()}"

    # --- Gemini's written explanation ---
    report_text = result.get("gemini_explanation", "No report available.")

    # --- Gallery of the retrieved historical cases, so the radiologist can
    #     visually verify the evidence themselves rather than trust a label ---
    gallery_items = [
        (m["path"], f"{m['label']} — similarity {m['similarity_score']}")
        for m in result["top_matches"]
    ]

    # --- Zero-shot cross-check as a simple table ---
    zero_shot_rows = [
        [s["label"], s["score"]] for s in result["zero_shot_check"]["scores"]
    ]

    return diagnosis_text, report_text, gallery_items, zero_shot_rows


with gr.Blocks(title="Brain Tumor RAG - Clinical Decision Support") as demo:
    gr.Markdown("# 🧠 Brain Tumor RAG Dashboard")
    gr.Markdown(
        "Upload a brain MRI scan. The system retrieves visually similar "
        "historical cases and generates an evidence-grounded explanation. "
        "**This is a decision-support aid, not a diagnostic tool.**"
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload MRI Scan")
            analyze_button = gr.Button("Analyze", variant="primary")

        with gr.Column(scale=2):
            diagnosis_output = gr.Markdown(label="Diagnosis")
            report_output = gr.Markdown(label="Gemini Explanation")

    gr.Markdown("### Retrieved Historical Cases")
    gallery_output = gr.Gallery(label="Top matches", columns=5, height=220)

    gr.Markdown("### Zero-Shot Cross-Check")
    zero_shot_output = gr.Dataframe(
        headers=["Label", "Score"],
        label="Independent text-based classification",
    )

    analyze_button.click(
        fn=analyze_scan,
        inputs=[image_input],
        outputs=[diagnosis_output, report_output, gallery_output, zero_shot_output],
    )


if __name__ == "__main__":
    demo.launch()