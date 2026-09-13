"""
Generative reasoning layer using Gemini.

Takes the uploaded MRI image PLUS the evidence we already retrieved from
Qdrant (and the zero-shot cross-check), and asks Gemini to write a short,
evidence-grounded explanation - the "G" in RAG. Gemini is NOT asked to
diagnose from scratch; it's asked to reason about evidence we already found,
which is what keeps this system "accountable" rather than a black box.
"""
from PIL import Image
from google import genai
import time

from config import GOOGLE_API_KEY, GEMINI_MODEL

_client = None  # module-level singleton, same idea as BioMedCLIPEmbedder._instance


def get_client() -> genai.Client:
    """Reuse a single Gemini client instead of creating a new one per request."""
    global _client
    if _client is None:
        if not GOOGLE_API_KEY:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Add it to your .env file "
                "(get a free key at https://aistudio.google.com/apikey)."
            )
        _client = genai.Client(api_key=GOOGLE_API_KEY)
    return _client


def build_prompt(primary_diagnosis: str, matches: list[dict], zero_shot_result: dict) -> str:
    """
    Turns our retrieval results into a plain-text description Gemini can
    reason over. We hand Gemini the EVIDENCE, not just an instruction to
    guess - this is what makes the response grounded rather than invented.
    """
    evidence_lines = "\n".join(
        f"  - {m['filename']}: labeled '{m['label']}', "
        f"similarity score {m['similarity_score']} (1.0 = identical)"
        for m in matches
    )

    zero_shot_lines = "\n".join(
        f"  - {s['label']}: {s['score']}" for s in zero_shot_result["scores"]
    )

    return f"""You are assisting a radiologist reviewing a brain MRI scan. You are
NOT making a final diagnosis - you are explaining the evidence a retrieval
system has already gathered, so the radiologist can verify it themselves.

RETRIEVED EVIDENCE
The uploaded scan was compared against a database of confirmed historical
cases. The {len(matches)} most visually similar cases found were:
{evidence_lines}

Top label by average similarity among these retrieved cases: {primary_diagnosis}

INDEPENDENT CROSS-CHECK
A separate zero-shot text-matching check (comparing the image directly
against class descriptions, without using the case database) gave these
scores:
{zero_shot_lines}

YOUR TASK
Write a short (3-5 sentence) evidence-based explanation for the radiologist.
Specifically:
1. State what the retrieved evidence suggests, and how strong that evidence
   is (based on the similarity scores - scores above ~0.98 indicate very
   close matches; noticeably lower or spread-out scores indicate a less
   clear-cut case).
2. Note whether the independent zero-shot check agrees or disagrees with the
   retrieval result, and what that means for confidence.
3. Do NOT invent clinical details you cannot see. Do NOT state a diagnosis as
   fact - describe it as what the evidence indicates.
4. End with a brief reminder that this is a decision-support aid, and the
   retrieved comparison images should be reviewed directly by the
   radiologist before any clinical decision.
"""


def generate_report(image: Image.Image, primary_diagnosis: str, matches: list[dict], zero_shot_result: dict) -> str:
    """
    Sends the image + evidence summary to Gemini and returns its written
    explanation as plain text.
    """
    client = get_client()
    prompt = build_prompt(primary_diagnosis, matches, zero_shot_result)

    # google-genai accepts a PIL Image directly alongside a text string in
    # the same `contents` list - no manual base64 encoding needed.
    # Retry a couple times on transient errors (e.g. 503 "high demand"),
    # since these usually clear up within a few seconds.
    last_error = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[image, prompt],
            )
            return response.text
        except Exception as e:
            last_error = e
            if attempt < 2:
                time.sleep(2 * (attempt + 1))  # wait 2s, then 4s, before retrying
    raise last_error