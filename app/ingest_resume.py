from endee import Endee
from pypdf import PdfReader
import requests
import os
import re



client = Endee()
index = client.get_index("resume_index")

print("Using index:", index.name)

RESUME_FOLDER = os.path.join(os.path.dirname(__file__), "data", "resumes")
  # folder with multiple resumes


def read_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + " "
    return text


def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r'[^A-Za-z0-9,.()@:/\- ]+', ' ', text)
    text = " ".join(text.split())
    return text


def chunk_text(text, size=600, overlap=120):
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + size]
        if len(chunk.strip()) > 50:   # avoid junk tiny chunks
            chunks.append(chunk)
        start += (size - overlap)
    return chunks


def detect_section(chunk):
    c = chunk.lower()
    if "education" in c:
        return "education"
    if "project" in c:
        return "projects"
    if "experience" in c or "intern" in c:
        return "experience"
    if "skill" in c or "technology" in c:
        return "skills"
    return "other"


def get_embedding(text):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": "nomic-embed-text",
            "prompt": text
        }
    )
    return response.json()["embedding"]


# -------- MULTI RESUME INGESTION --------

for file in os.listdir(RESUME_FOLDER):

    if file.endswith(".pdf"):

        path = os.path.join(RESUME_FOLDER, file)
        print("Processing:", file)

        raw_text = read_pdf(path)
        text = clean_text(raw_text)

        chunks = chunk_text(text)
        print(f"{file} -> {len(chunks)} chunks created")


        role_tag = file.split("_")[0]   # sde_resume.pdf -> sde

        for i, chunk in enumerate(chunks):

            vector = get_embedding(chunk)
            section = detect_section(chunk)

            index.upsert([
                {
                    "id": f"{file}_chunk_{i}",   # IMPORTANT: prevents overwrite
                    "vector": vector,
                    "meta": {
                        "text": chunk,
                        "source": file,
                        "role": role_tag,
                        "section": section
                    }
                }
            ])

print("All resumes indexed successfully with Ollama + Endee")
