import os
from pathlib import Path
from playwright.async_api import async_playwright

PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def html_to_pdf(html_content: str, rfp_id: str, filename: str) -> str:
    """
    Convert HTML content to PDF using Playwright and save it inside
    project_root/quotation/<rfp_id>/ folder.
    """
    # Ensure quotation folder exists
    rfp_folder = PROJECT_ROOT / "quotation" / rfp_id
    rfp_folder.mkdir(parents=True, exist_ok=True)

    output_path = rfp_folder / filename

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.set_content(html_content, wait_until="networkidle")
        await page.pdf(
            path=str(output_path),
            format="A4",
            print_background=True,
        )
        await browser.close()
        return str(output_path.resolve())


def pdf_to_bytes(rfp_id: str, filename: str) -> bytes:
    """
    Return PDF file bytes for a given RFP ID and filename.
    """
    pdf_path = PROJECT_ROOT / "quotation" / rfp_id / filename
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    with open(pdf_path, "rb") as f:
        return f.read()