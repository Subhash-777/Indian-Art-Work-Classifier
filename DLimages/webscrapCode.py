import os, requests, io, csv, time, random
from PIL import Image
from bs4 import BeautifulSoup

# Root folder
ROOT = "indian_art_dataset_5000"
os.makedirs(ROOT, exist_ok=True)

# Define classes and queries
CLASSES = {
    "kerala_mural_painting_kerala": "Kerala Mural Painting",
    "kondapalli_bommallu_andhra_pradesh": "Kondapalli Bommallu Andhra",
    "kutch_lippan_art_gujarat": "Kutch Lippan Art Gujarat",
    "leather_puppet_art_andhra_pradesh": "Leather Puppet Art Andhra",
    "madhubani_painting_bihar": "Madhubani Painting Bihar",
    "mandala_art": "Mandala Art",
    "mandana_art_rajasthan": "Mandana Art Rajasthan",
    "mata_ni_pachedi_gujarat": "Mata Ni Pachedi Gujarat",
    "meenakari_painting_rajasthan": "Meenakari Painting Rajasthan",
    "mughal_paintings": "Mughal Paintings",
    "mysore_ganjifa_art_karnataka": "Mysore Ganjifa Art Karnataka",
    "pattachitra_painting_odisha_bengal": "Pattachitra Painting Odisha Bengal",
    "patua_painting_west_bengal": "Patua Painting West Bengal",
    "pichwai_painting_rajasthan": "Pichwai Painting Rajasthan",
    "rajasthani_miniature_painting_rajasthan": "Rajasthani Miniature Painting Rajasthan",
    "rogan_art_kutch_gujarat": "Rogan Art Kutch Gujarat",
    "sohrai_art_jharkhand": "Sohrai Art Jharkhand",
    "tikuli_art_bihar": "Tikuli Art Bihar",
    "warli_folk_painting_maharashtra": "Warli Painting Maharashtra"
}

HEADERS = {"User-Agent": "Mozilla/5.0"}

# Function: download and save image
def download_image(url, folder, idx):
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        filename = os.path.join(folder, f"{idx:05d}.jpg")
        if not os.path.exists(filename):  # avoid overwriting
            img.save(filename, "JPEG", quality=90)
        return True
    except Exception as e:
        print(f"⚠️ Failed to download {url}: {e}")
        return False

# Function: fetch page with retries
def fetch_html(search_url, params):
    for attempt in range(10):  # retry up to 10 times
        try:
            html = requests.get(search_url, params=params, headers=HEADERS, timeout=30)
            html.raise_for_status()
            return html.text
        except Exception as e:
            wait_time = 10 * (attempt + 1) + random.uniform(0, 5)  # slower backoff
            print(f"⚠️ Attempt {attempt+1} failed: {e}, retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)
    return None

# Function: scrape Bing Images
def scrape_bing(query, folder, csv_path, max_images=500):
    os.makedirs(folder, exist_ok=True)

    # resume support: find already downloaded count
    existing_files = sorted([f for f in os.listdir(folder) if f.endswith(".jpg")])
    count = len(existing_files)
    page = count // 50  # resume from approximate page

    # open CSV (append if exists, else write header)
    mode = "a" if os.path.exists(csv_path) else "w"
    with open(csv_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if mode == "w":
            writer.writerow(["filename", "url", "query"])

        while count < max_images:
            search_url = "https://www.bing.com/images/async"
            params = {
                "q": query,
                "first": str(page * 50),
                "count": "50",
                "relp": "50",
                "scenario": "ImageBasicHover"
            }

            html_text = fetch_html(search_url, params)
            if not html_text:
                print("⚠️ No HTML returned, stopping.")
                break

            soup = BeautifulSoup(html_text, "html.parser")
            images = soup.find_all("img", {"class": "mimg"})
            if not images:
                print("⚠️ No more images found, stopping.")
                break

            for img in images:
                url = img.get("src") or img.get("data-src")
                if not url:
                    continue
                if download_image(url, folder, count):
                    writer.writerow([f"{count:05d}.jpg", url, query])
                    count += 1
                    if count >= max_images:
                        break

            page += 1
            time.sleep(2 + random.uniform(0, 2))  # polite delay + jitter

    print(f"✅ {count} images downloaded for '{query}'")

# Loop through all classes
for cls, query in CLASSES.items():
    folder = os.path.join(ROOT, cls, "images")
    csv_path = os.path.join(ROOT, cls, "metadata.csv")
    scrape_bing(query, folder, csv_path, max_images=5000)  # adjust max_images as needed
