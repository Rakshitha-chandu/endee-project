from endee import Endee
import requests

client = Endee()
index = client.get_index("resume_index")

def get_embedding(text):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": "nomic-embed-text",
            "prompt": text
        }
    )
    return response.json()["embedding"]


def retrieve_relevant(query, role_filter=None):

    vec = get_embedding(query)

    results = index.query(
        vector=vec,
        top_k=8   # increased for better sorting
    )

    projects = []
    experience = []
    skills = []
    education = []
    other = []

    for r in results:

        meta = r.get("meta", {})
        text = meta.get("text", "").strip()
        section = meta.get("section", "other")
        role = meta.get("role", None)

        # Optional role-based filtering
        if role_filter and role and role_filter != role:
            continue

        if not text:
            continue

        if section == "projects":
            projects.append(text)
        elif section == "experience":
            experience.append(text)
        elif section == "skills":
            skills.append(text)
        elif section == "education":
            education.append(text)
        else:
            other.append(text)

    # PRIORITY ORDER (strongest first)
    snippets = (
        projects[:2] +
        experience[:2] +
        skills[:1] +
        education[:1] +
        other[:1]
    )

    return snippets
