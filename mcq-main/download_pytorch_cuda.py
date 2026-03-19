"""Download PyTorch CUDA wheel with resume support to handle connection resets."""
import os
import sys
import time
import urllib.request

URL = "https://download.pytorch.org/whl/cu126/torch-2.10.0%2Bcu126-cp314-cp314-win_amd64.whl"
FILENAME = "torch-2.10.0+cu126-cp314-cp314-win_amd64.whl"
CHUNK_SIZE = 1024 * 1024  # 1MB chunks

def download_with_resume():
    downloaded = 0
    if os.path.exists(FILENAME):
        downloaded = os.path.getsize(FILENAME)
        print(f"Resuming from {downloaded / 1e9:.2f} GB...")

    max_retries = 10
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(URL)
            if downloaded > 0:
                req.add_header("Range", f"bytes={downloaded}-")

            with urllib.request.urlopen(req, timeout=120) as resp:
                total = int(resp.headers.get("Content-Length", 0)) + downloaded
                print(f"Total size: {total / 1e9:.2f} GB")

                with open(FILENAME, "ab" if downloaded > 0 else "wb") as f:
                    while True:
                        chunk = resp.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        pct = downloaded / total * 100 if total else 0
                        print(f"\r  {downloaded / 1e9:.2f} / {total / 1e9:.2f} GB ({pct:.1f}%)", end="", flush=True)

            print(f"\n\nDownload complete: {FILENAME}")
            return True

        except Exception as e:
            downloaded = os.path.getsize(FILENAME) if os.path.exists(FILENAME) else 0
            print(f"\n\nConnection lost at {downloaded / 1e9:.2f} GB (attempt {attempt + 1}/{max_retries})")
            print(f"  Error: {e}")
            print(f"  Retrying in 3 seconds...")
            time.sleep(3)

    print("Failed after all retries.")
    return False


if __name__ == "__main__":
    if download_with_resume():
        print(f"\nInstalling...")
        os.system(f"{sys.executable} -m pip install --force-reinstall {FILENAME}")
        print("\nCleaning up wheel file...")
        os.remove(FILENAME)
        print("Done! Run: python -c \"import torch; print(torch.cuda.is_available())\"")
    else:
        print("Download failed. Please try again or check your internet connection.")
