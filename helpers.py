import re
from skill_taxonomy import SKILL_TAXONOMY
from datetime import datetime

STOPWORDS = {
    "the","and","a","an","of","in","on","for","to","is","are",
    "with","by","at","as","from","that","this","it","be","or",
    "we","our","you","your","their","they","i"
}

SINGLE_EXP_RE = re.compile(
    r"""
    (?:
        over\s+|
        more\s+than\s+|
        at\s+least\s+|
        minimum\s+|
        approx(?:imately)?\s*
    )?
    (\d{1,2})
    \s*
    \+?
    \s*
    (?:years?|yrs?)
    """,
    re.IGNORECASE | re.VERBOSE
)

RANGE_EXP_RE = re.compile(
    r"""
    (\d{1,2})
    \s*
    (?:-|–|—|to)
    \s*
    (\d{1,2})
    \s*
    (?:years?|yrs?)
    """,
    re.IGNORECASE | re.VERBOSE
)

DATE_RANGE_RE = re.compile(
    r"""
    (?P<start>
        (?:jan|feb|mar|apr|may|jun|
         jul|aug|sep|sept|oct|nov|dec)?
        \s*
        (?:19|20)\d{2}
    )
    \s*
    (?:-|–|—|to)
    \s*
    (?P<end>
        (?:jan|feb|mar|apr|may|jun|
         jul|aug|sep|sept|oct|nov|dec)?
        \s*
        (?:19|20)\d{2}
        |present|current
    )
    """,
    re.IGNORECASE | re.VERBOSE
)
 
def tokenize(text: str):
    tokens = re.findall(r"[a-zA-Z+]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]

def infer_experience(resume_text: str):
    years, level = _infer_exp_explicit(resume_text)
    if years is not None:
        return years, level

    years = infer_experience_from_dates(resume_text)
    if years is not None:
        level = (
            "E1" if years <= 3 else
            "E2" if years <= 7 else
            "E3"
        )
        return years, level

    return None, "Not specified"

def _infer_exp_explicit(resume_text: str):
    text = resume_text.lower()

    range_match = RANGE_EXP_RE.search(text)
    if range_match:
        low = int(range_match.group(1))
        high = int(range_match.group(2))
        years = (low + high) // 2
    else:
        single_match = SINGLE_EXP_RE.search(text)
        if not single_match:
            return None, None
        years = int(single_match.group(1))

    if years <= 3:
        level = "E1"
    elif years <= 7:
        level = "E2"
    else:
        level = "E3"

    return years, level

def infer_jd_experience_range(jd_text: str):
    text = jd_text.lower()

    range_match = RANGE_EXP_RE.search(text)
    if range_match:
        return int(range_match.group(1)), int(range_match.group(2))

    single_match = SINGLE_EXP_RE.search(text)
    if single_match:
        yrs = int(single_match.group(1))
        return yrs, yrs

    return None, None

def infer_experience_from_dates(resume_text: str):
    text = resume_text.lower()
    matches = DATE_RANGE_RE.findall(text)

    if not matches:
        return None

    current_year = datetime.now().year
    spans = []

    for start_raw, end_raw in matches:
        try:
            start_year = int(re.search(r"(19|20)\d{2}", start_raw).group())
            if re.search(r"present|current", end_raw):
                end_year = current_year
            else:
                end_year = int(re.search(r"(19|20)\d{2}", end_raw).group())

            if end_year >= start_year:
                spans.append((start_year, end_year))
        except Exception:
            continue

    if not spans:
        return None

    spans.sort(key=lambda x: x[0])

    merged = [spans[0]]
    for s, e in spans[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e + 1:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))

    total_years = sum(e - s for s, e in merged)

    return total_years if total_years > 0 else None


def build_detailed_reason(matched, missing, sem_score, experience):
    parts = []

    if matched:
        parts.append(f"Matched {len(matched)} JD skills: {', '.join(matched[:6])}")
    else:
        parts.append("No strong JD skill matches")

    if missing:
        parts.append(f"Missing skills: {', '.join(missing[:4])}")

    if experience:
        parts.append(f"Experience: {experience}")
    else:
        parts.append("Experience not specified")

    if sem_score >= 0.75:
        parts.append("Very strong semantic alignment with JD responsibilities")
    elif sem_score >= 0.55:
        parts.append("Moderate semantic alignment with JD responsibilities")
    else:
        parts.append("Weak semantic alignment with JD responsibilities")

    return ". ".join(parts) + "."

def assign_decisions(results, jd_exp_range):
    total = len(results)
    consider_cutoff = int(0.2 * total)
    maybe_cutoff = int(0.5 * total)

    rank_map = {r["name"]: i for i, r in enumerate(results)}

    for r in results:
        decision = "Reject"
        score = r["score"]
        rank = rank_map[r["name"]]

        jd_min, jd_max = jd_exp_range
        years = r.get("years_experience")

        if jd_min is not None and jd_max is not None:
            if years is None:
                r["reason"] += " Experience not specified in resume."
                r["decision"] = "Reject"
                continue

            if years < jd_min or years > jd_max:
                r["reason"] += f" Experience outside JD range ({jd_min}-{jd_max} years)."
                r["decision"] = "Reject"
                continue

        if rank < consider_cutoff:
            decision = "Consider"
        elif rank < maybe_cutoff:
            decision = "Maybe"

        if score < 50:
            decision = "Reject"

        r["decision"] = decision

def two_phase_rank(results, top_k=10):
    """
    Phase 1: Semantic shortlist
    Phase 2: Human-aligned rerank
    Phase 3: Append remaining candidates
    """

    if not results:
        return results

    # Phase 1: semantic sort (global)
    semantic_sorted = sorted(
        results,
        key=lambda r: r.get("semantic_score", 0.0),
        reverse=True
    )

    # Phase 2: rerank only top K
    top_semantic = semantic_sorted[:top_k]

    reranked_top = sorted(
        top_semantic,
        key=lambda r: (
            r.get("skill_coverage", 0.0),
            r.get("years_experience") or 0,
            r.get("semantic_score", 0.0),
        ),
        reverse=True
    )

    # Phase 3: append the rest (unchanged order)
    remainder = semantic_sorted[top_k:]

    return reranked_top + remainder


def extract_skills_from_taxonomy(text:str):
    text = text.lower()
    found = set()

    for key,canonical in SKILL_TAXONOMY.items():
        if re.search(rf"\b{re.escape(key)}\b",text):
            found.add(canonical)
    return sorted(found)
