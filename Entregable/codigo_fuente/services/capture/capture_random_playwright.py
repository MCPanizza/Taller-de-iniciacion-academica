"""
Script de captura de sitios RANDOM SEGUROS para dataset de control.
Usa lista blanca de dominios confiables.
"""
import asyncio
import csv
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urlparse

from PIL import Image
import imagehash
from playwright.async_api import TimeoutError as PWTimeout, async_playwright

ROOT = Path(__file__).resolve().parents[2]
INPUT_TXT = ROOT / "docs" / "safe_urls.txt"
OUTPUT_DIR = ROOT / "datasets" / "mini" / "screenshots_random"
MANIFEST_PATH = ROOT / "datasets" / "mini" / "random_manifest.csv"
VIEWPORT = {"width": 800, "height": 600}
TIMEOUT = 30_000
WAIT_AFTER_LOAD_MS = 2_000
RETRIES = 2

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_slug_cleanup_re = re.compile(r"[^a-z0-9]+")

def slugify(value: str) -> str:
    normalized = value.lower()
    slug = _slug_cleanup_re.sub("-", normalized).strip("-")
    return slug[:50] or "site"

def compute_phash(image_path: Path) -> str:
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            return str(imagehash.phash(img))
    except Exception:
        return ""

def extract_domain(url: str) -> str:
    try:
        parsed = urlparse(url)
        return parsed.netloc.replace("www.", "")
    except Exception:
        return "unknown"

async def capture_url(context, url: str) -> Tuple[bool, Optional[str], Optional[int], Optional[str]]:
    domain = extract_domain(url)
    slug = slugify(domain)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    record_id = f"random_{slug}_{timestamp}"
    target_path = OUTPUT_DIR / f"{record_id}.png"
    
    page = await context.new_page()
    status_code = None
    
    try:
        response = await page.goto(url, wait_until="domcontentloaded", timeout=TIMEOUT)
        if response:
            status_code = response.status
        
        await page.wait_for_timeout(WAIT_AFTER_LOAD_MS)
        
        await page.screenshot(path=str(target_path), full_page=False)
        await page.close()
        return True, record_id, status_code, None
        
    except Exception as err:
        await page.close()
        return False, record_id, status_code, str(err)

async def capture_all(urls: List[str]) -> List[dict]:
    results = []
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(viewport=VIEWPORT)
        
        for idx, url in enumerate(urls, start=1):
            print(f"[{idx}/{len(urls)}] Capturando: {url}...")
            
            success = False
            record_id = None
            status_code = None
            error = None
            
            for _ in range(RETRIES):
                success, record_id, status_code, error = await capture_url(context, url)
                if success:
                    break
            
            phash_value = ""
            image_rel_path = ""
            if success and record_id:
                image_path = OUTPUT_DIR / f"{record_id}.png"
                if image_path.exists():
                    image_rel_path = str(image_path.relative_to(ROOT))
                    phash_value = compute_phash(image_path)
            
            results.append({
                "id": record_id or f"error_{idx}",
                "url": url,
                "label": "random_legit",  # Nueva etiqueta segura
                "bank": "unknown",
                "brand_state": "unknown",
                "country": "global",
                "lang": "en",
                "ts_iso": datetime.now().isoformat(),
                "viewport": f"{VIEWPORT['width']}x{VIEWPORT['height']}",
                "phash": phash_value,
                "source": "safe_list",
                "http_status": status_code or "",
                "image_path": image_rel_path,
                "notes": error or "",
            })
            
            if success:
                print(f"  [OK] {record_id}")
            else:
                print(f"  [ERROR] {error}")
                
        await context.close()
        await browser.close()
    return results

def load_urls() -> List[str]:
    if not INPUT_TXT.exists():
        return []
    with INPUT_TXT.open("r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

def write_manifest(rows: List[dict]):
    fieldnames = ["id", "url", "label", "bank", "brand_state", "country", "lang", 
                  "ts_iso", "viewport", "phash", "source", "http_status", "image_path", "notes"]
    
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nManifest guardado: {MANIFEST_PATH}")

if __name__ == "__main__":
    urls = load_urls()
    if urls:
        print(f"Iniciando captura de {len(urls)} sitios seguros...")
        rows = asyncio.run(capture_all(urls))
        write_manifest(rows)
    else:
        print("No se encontraron URLs en safe_urls.txt")

