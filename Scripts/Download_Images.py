import os
import random
import requests
from bs4 import BeautifulSoup

url = "https://cdn.star.nesdis.noaa.gov/GOES16/ABI/SECTOR/smv/03/"

save_dir = "Water_Vapor_Images"
os.makedirs(save_dir, exist_ok=True)

def download_image(img_url, save_path):
    try:
        response = requests.get(img_url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            print(f"Downloaded {save_path}")
        else:
            print(f"Failed to download {img_url} - Status code: {response.status_code}")
    except Exception as e:
        print(f"Error downloading {img_url}: {e}")

try:
    response = requests.get(url)
    response.raise_for_status()  # Ensure we got a valid response
    soup = BeautifulSoup(response.text, "html.parser")

    image_links = [link.get("href") for link in soup.find_all("a") if link.get("href") and "600x600" in link.get("href")]

    selected_links = random.sample(image_links, min(100, len(image_links)))

    for href in selected_links:
        img_url = url + href
        img_name = href.split("/")[-1]
        save_path = os.path.join(save_dir, img_name)
        download_image(img_url, save_path)

except Exception as e:
    print(f"Error accessing {url}: {e}")
