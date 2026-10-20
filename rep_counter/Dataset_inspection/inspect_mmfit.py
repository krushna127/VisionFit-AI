
from pathlib import Path
from collections import Counter

# Change this path to your actual MM-Fit dataset folder.
DATASET_DIR = Path(r"C:\Users\Krushna\Desktop\VisionFit-AI\dataset\mmfit")

if not DATASET_DIR.exists():
    print("Dataset folder not found!")
    print("Update DATASET_DIR with the correct folder path.")
else:
    files = [p for p in DATASET_DIR.rglob("*") if p.is_file()]

    print(f"Dataset folder: {DATASET_DIR}")
    print(f"Total files: {len(files)}")
    print("\nFile types:")

    extensions = Counter(
        p.suffix.lower() or "[no extension]" for p in files
    )

    for extension, count in extensions.most_common():
        print(f"{extension}: {count}")

    print("\nFirst 40 files:")
    for path in files[:40]:
        print(path.relative_to(DATASET_DIR))
