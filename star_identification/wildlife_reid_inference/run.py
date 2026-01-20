"""
Main script to build database and launch lineup app
"""
import os
import sys
from pathlib import Path

# Ensure we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from wildlife_reid_inference.run_lineup_tools import build_database, launch_app

# Configuration - paths are now relative to wildlife_reid_inference folder
REID_CHECKPOINT = Path(__file__).parent / "best_model.pth"
YOLO_CHECKPOINT = Path(__file__).parent / "starseg_best.pt"
DATA_DIR = Path(__file__).parent.parent / "archive" / "sequence_sorted_r1"
OUTPUT_DB = Path(__file__).parent / "lineup_db.npz"

def main():
    print("Wildlife ReID Database Builder & Launcher")
    print("=" * 60)

    # Convert paths to strings for compatibility
    reid_checkpoint = str(REID_CHECKPOINT)
    yolo_checkpoint = str(YOLO_CHECKPOINT)
    data_dir = str(DATA_DIR)
    output_db = str(OUTPUT_DB)

    # Step 1: Check for required files
    print("\n1. Checking required files:")

    missing_files = []

    # Check ReID checkpoint
    if not REID_CHECKPOINT.exists():
        print(f"   ✗ ReID checkpoint NOT FOUND: {reid_checkpoint}")
        missing_files.append(reid_checkpoint)
    else:
        print(f"   ✓ ReID checkpoint found: {reid_checkpoint}")

    # Check for config file
    checkpoint_dir = REID_CHECKPOINT.parent
    config_files = list(checkpoint_dir.glob("*config*.json"))

    if not config_files:
        print(f"   ✗ Config file NOT FOUND in {checkpoint_dir}")
        print(f"     You need to copy your training config JSON to this directory")
        missing_files.append("config.json")
    else:
        print(f"   ✓ Config file found: {config_files[0].name}")

    # Check YOLO checkpoint
    if not YOLO_CHECKPOINT.exists():
        print(f"   ✗ YOLO checkpoint NOT FOUND: {yolo_checkpoint}")
        missing_files.append(yolo_checkpoint)
    else:
        print(f"   ✓ YOLO checkpoint found: {yolo_checkpoint}")

    # Check data directory
    if not DATA_DIR.exists():
        print(f"   ✗ Data directory NOT FOUND: {data_dir}")
        missing_files.append(data_dir)
    else:
        print(f"   ✓ Data directory found: {data_dir}")
        subdirs = [d for d in DATA_DIR.iterdir() if d.is_dir()]
        print(f"     Contains {len(subdirs)} subdirectories")

    if missing_files:
        print(f"\n✗ Cannot proceed - missing required files!")
        if "config.json" in missing_files:
            print("\nTo fix the missing config:")
            print("1. Find your training output directory")
            print("2. Look for files like: pretrain_config.json, finetune_config.json")
            print("3. Copy the config file to wildlife_reid_inference/")
        sys.exit(1)

    # Step 2: Build database
    print(f"\n2. Building database:")
    print(f"   Source: {data_dir}")
    print(f"   Output: {output_db}")
    print(f"   This may take several minutes...\n")

    result = build_database(
        reid_checkpoint,
        yolo_checkpoint,
        data_dir,
        output_path=output_db
    )

    # Check result
    if result["status"] == "completed" and result.get("return_code", 1) == 0:
        if Path(output_db).exists():
            file_size = os.path.getsize(output_db) / 1e6
            print(f"\n✅ Database built successfully!")
            print(f"   File: {output_db}")
            print(f"   Size: {file_size:.2f} MB")

            # Step 3: Launch app
            print(f"\n3. Launching lineup application...")
            try:
                launch_app(reid_checkpoint, yolo_checkpoint, output_db)
                print("\n✅ Application closed")
            except Exception as e:
                print(f"\n✗ Error launching app: {e}")
        else:
            print(f"\n✗ Database file was not created!")
    else:
        print(f"\n✗ Database build failed!")
        print(f"   Status: {result.get('status', 'unknown')}")
        print(f"   Return code: {result.get('return_code', 'N/A')}")

        if result.get("stderr"):
            print("\nError output:")
            print("-" * 60)
            print(result["stderr"])

        if result.get("stdout"):
            print("\nStandard output (last 20 lines):")
            print("-" * 60)
            lines = result["stdout"].strip().split('\n')
            for line in lines[-20:]:
                print(line)

if __name__ == "__main__":
    main()