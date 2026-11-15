"""
Model downloader for PixelTalk.

Downloads the model files into the normal Hugging Face cache
(Windows: C:\\Users\\<you>\\.cache\\huggingface\\hub) WITHOUT loading them into memory,
so you can run this once while online and use the app afterwards.

Usage:
    python download_models.py                 # main VLM + baseline models  (~6 GB)
    python download_models.py --vlm-only      # only the main chat VLM       (~4.5 GB)
    python download_models.py --with-fallback # also the small BLIP-VQA fallback
    python download_models.py --vlm Qwen/Qwen2-VL-2B-Instruct
"""

import argparse
import sys

DEFAULT_VLM = "Qwen/Qwen2-VL-2B-Instruct"
IGNORE = ["*.msgpack", "*.h5", "*.onnx", "*.ot", "flax_model*", "tf_model*", "*.tflite"]


def main():
    parser = argparse.ArgumentParser(description="Download PixelTalk model weights")
    parser.add_argument("--vlm", default=DEFAULT_VLM, help=f"chat VLM to download (default {DEFAULT_VLM})")
    parser.add_argument("--vlm-only", action="store_true", help="skip the baseline models")
    parser.add_argument("--with-fallback", action="store_true", help="also download BLIP-VQA (fallback engine)")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[!] huggingface_hub is missing. Run: pip install -r requirements.txt")
        sys.exit(1)

    models = [("Main vision-language chat model", args.vlm)]
    if not args.vlm_only:
        models += [
            ("Baseline image captioner", "Salesforce/blip-image-captioning-base"),
            ("Baseline dialogue model", "google/flan-t5-small"),
        ]
    if args.with_fallback:
        models.append(("Fallback VQA model", "Salesforce/blip-vqa-base"))

    print("=" * 64)
    print("PixelTalk model downloader  (already-downloaded files are skipped)")
    print("=" * 64)
    failed = []
    for label, repo in models:
        print(f"\n[*] {label}: {repo}")
        try:
            path = snapshot_download(repo_id=repo, ignore_patterns=IGNORE)
            print(f"    [OK] {path}")
        except Exception as exc:  # noqa: BLE001
            failed.append(repo)
            print(f"    [!] Failed: {type(exc).__name__}: {exc}")
            print("        Check your internet connection and free disk space, then run this script again;")
            print("        it resumes where it stopped.")

    print("\n" + "=" * 64)
    if failed:
        print("Finished WITH ERRORS for: " + ", ".join(failed))
        sys.exit(1)
    print("All models are cached. Start the app with:  streamlit run app.py")
    print("=" * 64)


if __name__ == "__main__":
    main()
