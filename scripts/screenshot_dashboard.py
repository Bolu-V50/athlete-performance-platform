"""Capture dashboard screenshots for the README.

Run the app first:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "figures"
URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 1100}, device_scale_factor=2)
        page.goto(URL, wait_until="networkidle", timeout=90_000)
        # Streamlit renders over a websocket after the shell loads; wait for the
        # attention queue rather than a fixed sleep.
        page.wait_for_selector("text=Who needs attention", timeout=90_000)
        page.wait_for_timeout(3500)

        page.screenshot(path=str(OUT / "dashboard_top.png"))
        print("saved figures/dashboard_top.png")

        # Streamlit renders inside a virtualised scroll container, so
        # full_page=True captures only the viewport. A tall viewport is the
        # reliable way to get the whole page in one image.
        page.set_viewport_size({"width": 1600, "height": 3200})
        page.wait_for_timeout(3000)
        page.screenshot(path=str(OUT / "dashboard_full.png"))
        print("saved figures/dashboard_full.png")
        browser.close()


if __name__ == "__main__":
    main()
