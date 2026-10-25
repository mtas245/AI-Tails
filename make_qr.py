from __future__ import annotations

from pathlib import Path
import sys

import qrcode


def main():
    if len(sys.argv) != 2:
        print("Usage: python make_qr.py https://your-demo-url")
        raise SystemExit(1)

    url = sys.argv[1].strip()
    if not url.startswith(("http://", "https://")):
        raise SystemExit("Please provide a full http(s) URL.")

    output = Path("pitch_qr.png")

    image = qrcode.make(url)
    image.save(output)

    print(f"QR code saved: {output.resolve()}")
    print(f"URL: {url}")


if __name__ == "__main__":
    main()
