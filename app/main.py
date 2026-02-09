from scrape_job import scrape_job
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


print("\n=== Personalized Referral Email Generator (RAG) ===\n")

profile = load_profile()

if profile:
    print("Candidate:", profile.get("name", ""))
    print("Education:", profile.get("degree", ""), "-", profile.get("university", ""))
    print()

# ----- JD INPUT -----

choice = input("1 = Paste JD text | 2 = Enter Job URL: ")

if choice == "2":
    url = input("Enter job URL: ")
    job_text = scrape_job(url)
else:
    job_text = input("\nPaste Job Description:\n")

# ----- ROLE SELECTION (MULTI-RESUME SUPPORT) -----

print("\nSelect target role (optional):")
print("1 = SDE")
print("2 = AI/ML")
print("3 = Backend")
print("4 = General")
print("Press Enter to skip")

role_choice = input("Choice: ").strip()

role_map = {
    "1": "sde",
    "2": "ai",
    "3": "backend",
    "4": "general"
}

role_filter = role_map.get(role_choice, None)

# ----- RETRIEVAL -----

snippets = retrieve_relevant(job_text, role_filter=role_filter)

print("\n--- Retrieved Resume Context (Top Evidence) ---\n")

for i, s in enumerate(snippets, 1):
    print(f"{i}. {s[:220]}\n")

# ----- GENERATION -----

email = generate_email(job_text, snippets)

print("\n--- Generated Referral Email ---\n")
print(email)
