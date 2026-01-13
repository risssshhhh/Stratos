import json
import re
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

import ollama
from sklearn.metrics.pairwise import cosine_similarity

from pypdf import PdfReader
from docx import Document
import os
import hashlib
from helpers import extract_skills_from_taxonomy
from helpers import two_phase_rank

os.environ["OLLAMA_HOST"] = "http://pcz-dgpu01:11434"
os.environ["HF_EMBEDDING_MODEL"] = "sentence-transformers/all-MiniLM-L6-v2"
os.environ["LLM_MODEL"] = "gpt-oss:20b" #qwen3:20b

OLLAMA_HOST = os.environ["OLLAMA_HOST"]
HF_EMBEDDING_MODEL = os.environ["HF_EMBEDDING_MODEL"]
LLM_MODEL = os.environ["LLM_MODEL"]

# DETERMINISM CONTROLS
HYRE_CACHE = {}  # Cache HYRE per JD to avoid randomness across reloads
OLLAMA_OPTIONS = {
    "temperature": 0.0,   #decreases randomness
    "top_p": 1.0
}
EMBEDDING_CACHE = {}
HYRE_EMB_CACHE = {}

# Experience bands 
EXP_LEVEL_RANGES = {
    "E1": (1, 3),
    "E2": (4, 7),
    "E3": (8, 12)
}

def stable_hash(text: str) -> str:
    return hashlib.sha1(
        text.encode("utf-8", errors="ignore")
    ).hexdigest()

EXP_NUM_RE = re.compile(r"(\d{1,2})")

def normalize_years_exp(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        m = EXP_NUM_RE.search(x)
        if m:
            return float(m.group(1))
    return None

def extract_jd_min_experience(jd_text: str):
    m = re.search(r"(\d{1,2})\s*\+?\s*(years|yrs)", jd_text.lower())
    if m:
        return int(m.group(1))
    return None

# RESUME FILE PARSING
def extract_text_from_pdf(path: str) -> str:
    text = ""
    try:
        reader = PdfReader(path)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    except Exception as e:
        print("PDF parse error:", e)

    return " ".join(text.split())


def extract_text_from_docx(path: str) -> str:
    try:
        doc = Document(path)
        text = "\n".join(p.text for p in doc.paragraphs)
        return " ".join(text.split())
    except Exception as e:
        print("DOCX parse error:", e)
        return ""


def load_resume_file(path: str) -> str:
    path = path.lower()

    if path.endswith(".pdf"):
        return extract_text_from_pdf(path)

    elif path.endswith(".docx"):
        return extract_text_from_docx(path)

    elif path.endswith(".txt"):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return " ".join(f.read().split())
        except Exception:
            return ""

    else:
        raise ValueError("Unsupported resume format")

# UTILITY FUNCTIONS
def parse_skills_str(x):
    if isinstance(x, str):
        s = x.strip()[1:-1]
        return [t.strip().strip("'\"") for t in s.split(",") if t.strip()]
    return []


def cosine_sim_single(a: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return cosine_similarity(a.reshape(1, -1), matrix)[0]


def compute_skill_overlap(jd_text: str, df: pd.DataFrame) -> np.ndarray:
    jd_skills = set(extract_skills_from_taxonomy(jd_text))
    scores = []

    for skills in df["skills"]:
        skills = set(skills or [])
        overlap = len(skills & jd_skills)
        scores.append(overlap / max(len(skills), 1))

    return np.array(scores, dtype="float32")


# EMBEDDING HANDLING
def try_load_embedder(hf_model_name: str = None):
    try:
        print("Model has started loading")
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(hf_model_name or HF_EMBEDDING_MODEL)
        print ("Model is loaded")
        return embedder, True, None
    except Exception as e:
        return None, False, e


def get_embeddings(embedder, texts: List[str]) -> np.ndarray:
    return embedder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

def embed_texts_cached(
    embedder,
    texts: List[str]
) -> np.ndarray:

    embeddings = [None] * len(texts)
    to_compute = []
    to_compute_idx = []

    for i, text in enumerate(texts):
        key = stable_hash(text)
        if key in EMBEDDING_CACHE:
            embeddings[i] = EMBEDDING_CACHE[key]
        else:
            to_compute.append(text)
            to_compute_idx.append(i)

    if to_compute:
        new_embs = get_embeddings(embedder, to_compute)
        for idx, emb, text in zip(to_compute_idx, new_embs, to_compute):
            k = stable_hash(text)
            EMBEDDING_CACHE[k] = emb
            embeddings[idx] = emb

    return np.vstack(embeddings)

def prepare_resume_embeddings(
    embedder,
    df_resumes: pd.DataFrame
) -> np.ndarray:
    """
    Prepare embeddings for resume_text column.
    Called by UI.
    """
    texts = df_resumes["resume_text"].tolist()
    return embed_texts_cached(embedder, texts)

# HYRE GENERATION (IDEAL RESUME)
def generate_hyre_from_jd(jd_text: str, job_title: str = "") -> str:
    cache_key = stable_hash(jd_text)
    if cache_key in HYRE_CACHE:
        return HYRE_CACHE[cache_key]

    prompt = f"""
You are an expert hiring analyst and resume evaluator.
 
Your task is to convert the given Job Description (JD) into a
**Hypothetical Ideal Resume (HYRE)** that represents a *perfectly qualified*
candidate for this role.
 
This HYRE will be used as a **semantic reference document** for candidate
matching and ranking. Accuracy, coverage, and relevance are critical.
 
────────────────────
CORE PRINCIPLES
────────────────────
• This is NOT creative writing.
• This is NOT a summary of the JD.
• This is NOT a job posting rewrite.
• This is NOT aspirational or exaggerated.
 
The HYRE must reflect what a **real, strong, hire-ready candidate** would
credibly have on their resume.
 
Do NOT invent:
• Technologies not mentioned or clearly implied
• Skills outside the JD scope
• Experience levels not supported by the JD
• Soft skills unless explicitly stated
 
Do NOT include:
• Generic buzzwords
• Vague statements
• Hiring recommendations
• Opinions or judgments
 
────────────────────
FORMAT (STRICT)
────────────────────
 
Summary:
• 3–5 concise sentences
• Clearly describe:
  – The candidate’s professional role
  – Seniority level
  – Domain specialization
• Must directly reflect the responsibilities and expectations in the JD
 
Skills:
• Bullet list only
• Include ONLY:
  – Technical skills
  – Tools
  – Frameworks
  – Programming languages
  – Methodologies
• Prioritize skills explicitly mentioned in the JD
• Include strongly implied skills only if unavoidable
• No soft skills unless explicitly required
 
Experience:
• 2–4 roles or project descriptions
• Each role must map clearly to JD responsibilities
• Use responsibility-focused bullets (what the candidate did)
• Avoid achievements, metrics, or storytelling
• Mention years of experience ONLY if implied by the JD
 
Education:
• Minimal
• Include degree level or background only if stated or implied
• If not mentioned, default to:
  – Bachelor’s degree (B.E / B.Tech or equivalent)
 
────────────────────
STYLE CONSTRAINTS
────────────────────
• Length: 200–400 words
• Tone: factual, professional, resume-like
• Language: concise, neutral, deterministic
• No markdown headers other than the section titles above
• No emojis, no formatting symbols
 
────────────────────
INPUT
────────────────────
Job Title:
{job_title}
 
Job Description:
\"\"\"{jd_text}\"\"\"
 
────────────────────
OUTPUT
────────────────────
Produce ONLY the Hypothetical Ideal Resume following the exact structure
and rules above. Do not add explanations or commentary.

"""

    try:
        resp = ollama.generate(
            model=LLM_MODEL,
            prompt=prompt,
            options={
                "temperature": 0.0,
                "top_p": 1.0,
                "repeat_penalty": 1.0
                }
        )
        hyre = resp.get("response", "").strip()
    except Exception:
        hyre = f"Summary: {job_title}. Skills: {', '.join(jd_text.split()[:12])}"

    HYRE_CACHE[cache_key] = hyre
    return hyre

# EXPLANATION ENGINE 

def build_explanation(
    jd_text: str,
    skills,
    years_exp,
    sem_score: float,
    skill_score: float,
    hybrid_score: float,
    rank: int
) -> str:
    """
    Produces a deterministic, audit-quality explanation.

    IMPORTANT:
    - Explains *why* the candidate scored this way
    - Does NOT recommend hiring decisions
    - Policy decisions are handled in the UI layer
    """

    jd_lower = jd_text.lower()

    if isinstance(skills, str):
        skills = parse_skills_str(skills)

    matched = [s for s in skills if s.lower() in jd_lower]
    missing = [s for s in skills if s.lower() not in jd_lower]

    years_exp = normalize_years_exp(years_exp)

    exp_part = (
        f"{int(years_exp)} years of experience"
        if years_exp is not None else
        "experience not specified"
    )

    sem_desc = (
        "very strong alignment with JD responsibilities"
        if sem_score >= 0.70 else
        "moderate alignment with JD responsibilities"
        if sem_score >= 0.50 else
        "weak alignment with JD responsibilities"
    )

    return (
        f"Matched JD skills: {', '.join(matched[:6]) or 'none identified'}. "
        f"Missing JD skills: {', '.join(missing[:4]) or 'none identified'}. "
        f"Skill overlap score: {skill_score:.2f}. "
        f"Semantic similarity score: {sem_score:.2f}, indicating {sem_desc}. "
        f"Experience assessment: {exp_part}. "
        f"Final ranking may be adjusted using experience and skill sufficiency validation."
    )

# MAIN RANKING PIPELINE

def rank_candidates_for_jd(
    df_resumes: pd.DataFrame,
    resume_embeddings: Optional[np.ndarray],
    embedder,
    jd_text: str,
    job_title: str = "",
    alpha_sem: float = 0.8,
    alpha_skill: float = 0.2,
    top_k: int = 50
):
    """
    End-to-end ranking pipeline.
    This is the ONLY function the UI should call.
    """
    if embedder is None:
        return None, ("Embedding model not loaded. Please initialize model.")
    
    # Step 1: Generate ideal resume
    hyre = generate_hyre_from_jd(jd_text, job_title)

    # Step 2: Prepare embeddings if not provided
    if resume_embeddings is None:
        resume_embeddings = prepare_resume_embeddings(embedder, df_resumes)

    # Step 3: Semantic similarity
    if embedder is not None and resume_embeddings is not None:

        hyre_key = stable_hash(hyre)
        if hyre_key in HYRE_EMB_CACHE:
            hyre_emb = HYRE_EMB_CACHE[hyre_key]
        else:
            hyre_emb = get_embeddings(embedder, [hyre])[0]
            HYRE_EMB_CACHE[hyre_key] = hyre_emb

        raw_sem = cosine_sim_single(hyre_emb, resume_embeddings)

        # SentenceTransformer cosine similarity already in [0,1]
        sem_scores = np.clip(raw_sem, 0.0, 1.0)

        skill_scores = compute_skill_overlap(jd_text, df_resumes)
    else:
        return ("Model Failed. Please recheck the settings or contact programmers")

    # Step 3: Hybrid score
    hybrid_scores = alpha_sem * sem_scores + alpha_skill * skill_scores

    # Step 4: Assemble results
    results = df_resumes.copy()
    results["semantic_score"] = sem_scores
    results["skill_score"] = skill_scores
    results["hybrid_score"] = hybrid_scores

    # Step 3.5: Experience adjustment (penalty-based, deterministic)
    # Step 3.5: Experience adjustment (penalty-based, deterministic)

    exp_adjustment = np.zeros(len(results), dtype="float32")

    jd_min_exp = extract_jd_min_experience(jd_text)

    for i in range(len(results)):
        years_raw = results.iloc[i]["years_experience"]
        years = normalize_years_exp(years_raw)

        # Unknown experience → small penalty
        if years is None:
            exp_adjustment[i] = -0.05
            continue

        # Below absolute minimum
        if years < EXP_LEVEL_RANGES["E1"][0]:
            exp_adjustment[i] = -0.15

        # Junior for mid/senior roles
        elif years < EXP_LEVEL_RANGES["E2"][0]:
            exp_adjustment[i] = -0.08

        # JD-specific hard mismatch
        if jd_min_exp and years < jd_min_exp:
            exp_adjustment[i] -= 0.20

    results["hybrid_score"] = np.clip(
        results["hybrid_score"] + exp_adjustment, 0.0, 1.0
    )

    results = results.sort_values(
        "hybrid_score", ascending=False
    ).reset_index(drop=True)

    records = results.to_dict(orient="records")
    ranked = two_phase_rank(records, top_k=len(records))
    results = pd.DataFrame(ranked)

    reasons = []

    for rank, row in enumerate(results.itertuples(index=False), start=1):
        reason = build_explanation(
            jd_text=jd_text,
            skills=row.skills,
            years_exp=row.years_experience,
            sem_score=row.semantic_score,
            skill_score=row.skill_score,
            hybrid_score=row.hybrid_score,
            rank=rank
        )
        reasons.append(reason)

    results["reason"] = reasons

    if top_k is None or top_k <= 0:
        return results, hyre

    return results.head(top_k), hyre




# SMOKE TEST

if __name__ == "__main__":
    print("Backend loaded OK")
    print("LLM:", LLM_MODEL)
    print("Embedding:", HF_EMBEDDING_MODEL)
