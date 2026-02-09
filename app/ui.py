import streamlit as st
from retrieve import retrieve_relevant
from generate_email import generate_email
import json
import os

PROFILE_PATH = "app/profile.json"

def load_profile():
    if os.path.exists(PROFILE_PATH):
        with open(PROFILE_PATH, "r") as f:
            return json.load(f)
    return {}

profile = load_profile()

st.set_page_config(page_title="AI Referral Email Generator", layout="wide")

st.title("RAG-Based Personalized Referral Email Generator")

# Candidate Info
if profile:
    st.sidebar.header("Candidate")
    st.sidebar.write(profile.get("name", ""))
    st.sidebar.write(profile.get("degree", ""))
    st.sidebar.write(profile.get("university", ""))

# Job Description Input
st.header("Job Description")

job_text = st.text_area("Paste Job Description", height=200)

# Role Selection
st.header("Select Target Role")

role_option = st.selectbox(
    "Role",
    ["None", "SDE", "AI/ML", "Backend", "General"]
)

role_map = {
    "SDE": "sde",
    "AI/ML": "ai",
    "Backend": "backend",
    "General": "general",
    "None": None
}

role_filter = role_map.get(role_option)

# Generate Button
if st.button("Generate Referral Email"):

    if not job_text.strip():
        st.warning("Please paste a Job Description first.")
    else:
        with st.spinner("Retrieving relevant resume context..."):
            snippets = retrieve_relevant(job_text, role_filter=role_filter)

        st.subheader("Retrieved Resume Evidence")

        for s in snippets:
            st.info(s)

        with st.spinner("Generating grounded referral email..."):
            email = generate_email(job_text, snippets)

        st.subheader("Generated Referral Email")
        st.text_area("Email Output", email, height=300)
