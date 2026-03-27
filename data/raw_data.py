import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://physionet.org/files/propofol-anesthesia-dynamics/1.0/"
SAVE_DIR = "data"


def download_recursive(url, save_path):
    os.makedirs(save_path, exist_ok=True)

    r = requests.get(url)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    for link in soup.find_all("a"):
        href = link.get("href")

        if href in ["../", None]:
            continue

        full_url = urljoin(url, href)
        local_path = os.path.join(save_path, href)

        if href.endswith("/"):
            # Directory
            download_recursive(full_url, local_path)
        else:
            # File
            if os.path.exists(local_path):
                print(f"Skipping (exists): {local_path}")
                continue

            print(f"Downloading {full_url}")
            with requests.get(full_url, stream=True) as file_data:
                file_data.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in file_data.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)


download_recursive(BASE_URL, SAVE_DIR)