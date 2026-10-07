# main.py - SkillTwin AI Semantic Matching Engine & Career Simulator
from typing import List, Dict, Any
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize FastAPI application
app = FastAPI(
    title="SkillTwin AI Backend",
    description="Semantic Vector Matching Engine and 'What-If' Career Simulator",
    version="1.0.0",
)

# Load lightweight open-source embedding model for skill vectorization
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# ------------------------------------------------------------------
# Request & Response Schemas
# ------------------------------------------------------------------
class MatchRequest(BaseModel):
    candidate_skills: List[str] = Field(
        ..., example=["Python", "FastAPI", "REST APIs", "SQL"]
    )
    target_role_title: str = Field(..., example="Backend Developer")
    target_role_skills: List[str] = Field(
        ..., example=["Python", "FastAPI", "Docker", "PostgreSQL", "Microservices"]
    )


class SimulationRequest(BaseModel):
    current_skills: List[str] = Field(
        ..., example=["Python", "FastAPI", "REST APIs"]
    )
    target_role_skills: List[str] = Field(
        ..., example=["Python", "FastAPI", "Docker", "Kubernetes", "AWS"]
    )
    hypothetical_new_skills: List[str] = Field(
        ..., example=["Docker", "AWS Cloud Practitioner"]
    )


class MatchResponse(BaseModel):
    match_score_percentage: float
    semantic_overlap: Dict[str, Any]
    missing_skills: List[str]


class SimulationResponse(BaseModel):
    baseline_score: float
    simulated_score: float
    score_delta: float
    projected_match_percentage: float
    newly_covered_requirements: List[str]


# ------------------------------------------------------------------
# Helper Functions (Semantic Vector Engine)
# ------------------------------------------------------------------
def get_text_embedding(text: str) -> np.ndarray:
    """Generates a dense vector embedding for a given string."""
    return embedder.encode(text, convert_to_numpy=True)


def calculate_profile_similarity(
    candidate_skills: List[str], role_skills: List[str]
) -> float:
    """Calculates semantic vector similarity between candidate skills and role requirements."""
    if not candidate_skills or not role_skills:
        return 0.0

    candidate_str = ", ".join(candidate_skills)
    role_str = ", ".join(role_skills)

    c_vec = get_text_embedding(candidate_str).reshape(1, -1)
    r_vec = get_text_embedding(role_str).reshape(1, -1)

    similarity = cosine_similarity(c_vec, r_vec)[0][0]
    return float(np.clip(similarity * 100, 0, 100))


def extract_skill_gaps(
    candidate_skills: List[str], role_skills: List[str], similarity_threshold: float = 0.65
) -> List[str]:
    """Identifies skills required by the role that lack semantic coverage in candidate profile."""
    missing = []
    if not candidate_skills:
        return role_skills

    cand_embeddings = embedder.encode(candidate_skills, convert_to_numpy=True)

    for role_skill in role_skills:
        r_emb = embedder.encode(role_skill, convert_to_numpy=True).reshape(1, -1)
        sims = cosine_similarity(r_emb, cand_embeddings)[0]
        max_sim = np.max(sims)

        if max_sim < similarity_threshold:
            missing.append(role_skill)

    return missing


# ------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "online", "system": "SkillTwin AI Vector Engine"}


@app.post("/api/v1/match", response_model=MatchResponse)
def evaluate_match(payload: MatchRequest):
    """Evaluates semantic match between candidate skills and a target role."""
    score = calculate_profile_similarity(
        payload.candidate_skills, payload.target_role_skills
    )
    gaps = extract_skill_gaps(payload.candidate_skills, payload.target_role_skills)

    return MatchResponse(
        match_score_percentage=round(score, 2),
        semantic_overlap={
            "matched_skills_count": len(payload.target_role_skills) - len(gaps),
            "total_role_skills": len(payload.target_role_skills),
        },
        missing_skills=gaps,
    )


@app.post("/api/v1/simulate-what-if", response_model=SimulationResponse)
def simulate_career_path(payload: SimulationRequest):
    """Interactive Career 'What-If' Simulator: Measures match score gain upon upskilling."""
    baseline = calculate_profile_similarity(
        payload.current_skills, payload.target_role_skills
    )

    combined_skills = list(
        set(payload.current_skills + payload.hypothetical_new_skills)
    )
    simulated = calculate_profile_similarity(
        combined_skills, payload.target_role_skills
    )

    baseline_gaps = extract_skill_gaps(
        payload.current_skills, payload.target_role_skills
    )
    simulated_gaps = extract_skill_gaps(
        combined_skills, payload.target_role_skills
    )

    covered = [skill for skill in baseline_gaps if skill not in simulated_gaps]

    return SimulationResponse(
        baseline_score=round(baseline, 2),
        simulated_score=round(simulated, 2),
        score_delta=round(simulated - baseline, 2),
        projected_match_percentage=round(simulated, 2),
        newly_covered_requirements=covered,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
