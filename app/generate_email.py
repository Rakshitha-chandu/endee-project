import os
import json
from groq import Groq

# FIXED env variable name
client = Groq(api_key=os.getenv("YOUR_API_KEY"))

PROFILE_PATH = "app/profile.json"


def load_profile():
    if os.path.exists(PROFILE_PATH):
        with open(PROFILE_PATH, "r") as f:
            return json.load(f)
    return {}


def generate_email(job_text, snippets):

    context = "\n".join(snippets)
    profile = load_profile()

    name = profile.get("name", "")
    degree = profile.get("degree", "")
    university = profile.get("university", "")
    graduation_year = profile.get("graduation_year", "")
    email = profile.get("email", "")
    linkedin = profile.get("linkedin", "")
    github = profile.get("github", "")

    prompt = f"""
You are the job candidate writing a referral request email.

STRICT RULES:
- Use ONLY the information present in RESUME CONTEXT.
- Do NOT invent names, universities, companies, or projects.
- If any detail is missing, keep it general.
- Mention only technologies and projects present in context.
- Keep it concise (8–10 lines).
- Tone: sincere, professional, human.

CANDIDATE PROFILE:
Name: {name}
Education: {degree} at {university} ({graduation_year})

JOB DESCRIPTION:
{job_text}

RESUME CONTEXT:
{context}

Write the referral request email.

End the email with:

Best regards,
{name}
Email: {email}
LinkedIn: {linkedin}
GitHub: {github}
"""

    chat = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": "You write grounded professional referral emails using only provided context."
            },
            {"role": "user", "content": prompt}
        ]
    )

    return chat.choices[0].message.content
