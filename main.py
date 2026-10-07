import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="SkillTwin Job Finder", layout="wide")

# Sample Job Database (Can be connected to MongoDB / PostgreSQL)
JOB_DATABASE = [
    {
        "title": "Frontend Developer",
        "company": "TechCorp",
        "skills": "React JavaScript HTML CSS Redux TypeScript Tailwind",
    },
    {
        "title": "Backend Developer",
        "company": "DataWorks",
        "skills": "Python Node.js Express PostgreSQL REST API Docker MongoDB",
    },
    {
        "title": "Full Stack Engineer",
        "company": "CloudInnovations",
        "skills": "React Node.js Express MongoDB JavaScript TypeScript Docker AWS",
    },
    {
        "title": "Data Scientist",
        "company": "AI Insights",
        "skills": "Python Pandas Machine Learning SQL Data Analysis Scikit-Learn NumPy",
    },
    {
        "title": "DevOps Engineer",
        "company": "Infrastructure Hub",
        "skills": "Docker Kubernetes AWS CI/CD Linux Terraform Python Shell",
    },
]


def calculate_matches(user_skills, jobs):
    df_jobs = pd.DataFrame(jobs)

    # Combine user skills and job skills into corpus
    corpus = [user_skills] + df_jobs["skills"].tolist()

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Compute Cosine Similarity between user vector (index 0) and all job vectors
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    df_jobs["match_score"] = (similarity_scores * 100).round(2)
    return df_jobs.sort_values(by="match_score", ascending=False)


# App UI
st.title("🎯 SkillTwin Job Finder")
st.subheader("Match your transferable skill profile to open opportunities")

user_skills_input = st.text_area(
    "Enter your skills (separated by space or comma):",
    placeholder="e.g. React JavaScript Node.js MongoDB CSS",
    height=100,
)

if st.button("Find Matching Jobs"):
    if user_skills_input.strip():
        results = calculate_matches(user_skills_input, JOB_DATABASE)

        st.markdown("### Top SkillTwin Matches")
        for idx, row in results.iterrows():
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{row['title']}** at *{row['company']}*")
                    st.caption(f"Required Skills: {row['skills']}")
                with col2:
                    st.metric(label="Match Score", value=f"{row['match_score']}%")
                st.divider()
    else:
        st.warning("Please enter at least one skill to search.")
