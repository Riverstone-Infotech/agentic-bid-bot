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

    base_dir = PROJECT_ROOT / "quotation" / rfp_id
    pdf_path = base_dir / filename

    # If file with original name doesn't exist, try without '_update'
    if not pdf_path.exists() and "_update" in filename:
        alt_filename = filename.replace('_update', '')
        alt_pdf_path = base_dir / alt_filename
        if alt_pdf_path.exists():
            pdf_path = alt_pdf_path
        else:
            raise FileNotFoundError(f"PDF not found: {pdf_path} or {alt_pdf_path}")
    elif not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    with open(pdf_path, "rb") as f:
        return f.read()