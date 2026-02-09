import requests
from bs4 import BeautifulSoup

def scrape_job(url):

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")

    text = soup.get_text(separator=" ", strip=True)

    return text
