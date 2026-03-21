import pandas as pd
import requests
import os
from urllib.parse import urlparse
from tqdm import tqdm   # optional progress bar

# ┌───────────────────────────────────────┐
# │  Change these three lines as needed   │
# └───────────────────────────────────────┘
csv_path       = "Amazon Scraper Dataset.csv"             # ← your CSV file
url_column     = "Image"                 # ← column name that contains the links
save_folder    = "downloaded_images"         # ← where to save files

os.makedirs(save_folder, exist_ok=True)

df = pd.read_csv(csv_path)

def download_image(row):
    url = row[url_column]
    if pd.isna(url) or not isinstance(url, str) or not url.strip():
        return "skipped (empty)"

    try:
        # Try to get clean filename from URL
        path = urlparse(url).path
        ext = os.path.splitext(path)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']:
            ext = '.jpg'  # fallback

        # You can also use row index or other column as name
        filename = f"img_{row.name:05d}{ext}"
        save_path = os.path.join(save_folder, filename)

        r = requests.get(url, timeout=12)
        r.raise_for_status()

        with open(save_path, 'wb') as f:
            f.write(r.content)

        return f"OK → {filename}"
    except Exception as e:
        return f"FAIL → {str(e)[:60]}"

# ─── Run the downloads ────────────────────────────────
results = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading"):
    results.append(download_image(row))

df['download_status'] = results

# Optional: save report
df.to_csv("download_report.csv", index=False)

print(f"Finished. Check folder: {save_folder}")
print(df['download_status'].value_counts())