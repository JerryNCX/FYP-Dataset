import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path
import shutil

# 1. Load model (Added device check for speed)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

classes = ["cpu", "video card", "motherboard", "memory ram", "internal hard drive", "power supply", "computer case", "cpu cooler", "other electronic item"]

source_dir = Path("Amazon Images")
output_dir = Path("Amazon classified Dataset")

# 2. Check if source exists
if not source_dir.exists():
    print(f"❌ SOURCE DIRECTORY NOT FOUND: {source_dir}")
    exit()

# 3. Create folders
for cls in classes:
    (output_dir / cls).mkdir(parents=True, exist_ok=True)

# 4. Find images (Support multiple extensions)
extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.png']
image_files = []
for ext in extensions:
    image_files.extend(source_dir.glob(ext))

print(f"🚀 Found {len(image_files)} images. Starting classification...")

for img_path in image_files:
    try:
        image = Image.open(img_path).convert("RGB")
        inputs = processor(text=classes, images=image, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)

        probs = outputs.logits_per_image.softmax(dim=1)
        pred_idx = probs.argmax().item()
        pred_class = classes[pred_idx]

        dest = output_dir / pred_class / img_path.name
        shutil.copy(img_path, dest)
        print(f"✅ {img_path.name} -> {pred_class}")

    except Exception as e:
        print(f"❌ Error processing {img_path.name}: {e}")

print("\n✨ All done! Check your output folder.")