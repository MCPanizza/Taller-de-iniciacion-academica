import asyncio
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
import imagehash
from playwright.async_api import TimeoutError as PWTimeout, async_playwright

ROOT = Path(__file__).resolve().parents[2]
INPUT_JSON = ROOT / "login_gallery_index.json"
OUTPUT_DIR = ROOT / "datasets" / "mini" / "screenshots_legit"
MANIFEST_PATH = ROOT / "datasets" / "mini" / "legit_manifest.csv"
VIEWPORT = {"width": 800, "height": 600}
TIMEOUT = 30_000
RETRIES = 2
WAIT_AFTER_LOAD_MS = 2_000

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_slug_cleanup_re = re.compile(r"[^a-z0-9]+")

ACCEPT_TEXT_CUES = [
    "accept",
    "i agree",
    "aceptar",
    "acepto",
    "entendido",
    "continuar",
    "ok",
    "cerrar",
    "permitir",
]

ACCEPT_SELECTORS = [
    "button[id*='accept']",
    "button[id*='agree']",
    "button[aria-label*='accept']",
    "button[aria-label*='acept']",
    "button[aria-label*='agree']",
    "#onetrust-accept-btn-handler",
    "button[class*='accept']",
    "button[class*='Agree']",
]


def slugify(value: str) -> str:
    normalized = value.lower()
    for src, dst in {"Ã¡": "a", "Ã©": "e", "Ã­": "i", "Ã³": "o", "Ãº": "u", "Ã±": "n", "Ã§": "c"}.items():
        normalized = normalized.replace(src, dst)
    slug = _slug_cleanup_re.sub("-", normalized).strip("-")
    return slug or "item"


def hx_from_image(image_path: Path) -> str:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        return str(imagehash.phash(img))


def derive_brand_state(url: str) -> str:
    if not url:
        return "unknown"
    lower = url.lower()
    if "login" in lower or "signin" in lower or "acceso" in lower:
        return "login"
    if any(key in lower for key in ["home", "inicio", "principal"]):
        return "home"
    return "unknown"


async def try_accept_banners(page) -> None:
    # click known selectors
    for selector in ACCEPT_SELECTORS:
        try:
            await page.click(selector, timeout=500)
        except Exception:
            continue

    # click buttons by text cues
    for cue in ACCEPT_TEXT_CUES:
        try:
            button = page.get_by_role("button", name=re.compile(cue, re.IGNORECASE))
            await button.click(timeout=500)
        except Exception:
            continue


async def capture_entry(context, entry: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[int], Optional[str]]:
    url = entry.get("url")
    bank = entry.get("bank", "unknown")
    slug = slugify(bank)
    ts_iso = entry.get("captured_at")
    if ts_iso:
        try:
            ts = datetime.fromisoformat(ts_iso)
            ts_compact = ts.strftime("%Y%m%dT%H%M%S")
        except ValueError:
            ts_compact = "na"
    else:
        ts_compact = "na"

    record_id = f"legit_{slug}_{ts_compact}"
    target_path = OUTPUT_DIR / f"{record_id}.png"

    page = await context.new_page()
    status_code = None

    try:
        response = await page.goto(url, wait_until="domcontentloaded", timeout=TIMEOUT)
        if response:
            status_code = response.status
        await page.wait_for_timeout(WAIT_AFTER_LOAD_MS)
        await try_accept_banners(page)
        await page.wait_for_timeout(1000)
        await page.screenshot(path=str(target_path), full_page=False)
        return True, record_id, status_code, None
    except PWTimeout as err:
        return False, record_id, status_code, f"timeout:{err}"
    except Exception as err:
        return False, record_id, status_code, str(err)
    finally:
        await page.close()


async def run(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results = []
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=['--no-sandbox'])
        context = await browser.new_context(viewport=VIEWPORT, user_agent="phishvision-research/1.0")

        for entry in entries:
            success = False
            last_error = None
            status_code = None
            record_id = None

            for attempt in range(1, RETRIES + 1):
                success, record_id, status_code, last_error = await capture_entry(context, entry)
                if success:
                    break
                await asyncio.sleep(1)

            rows = build_rows(entry, record_id, success, status_code, last_error)
            if success:
                print(f"[OK] {entry.get('bank')} -> {record_id}")
            else:
                print(f"[FAIL] {entry.get('bank')} -> {last_error}")
            results.append(rows)

        await context.close()
        await browser.close()
    return results


def build_rows(entry: Dict[str, Any], record_id: Optional[str], success: bool, status_code: Optional[int], error: Optional[str]) -> Dict[str, Any]:
    image_rel_path = ""
    phash_value = ""
    if success and record_id:
        image_path = OUTPUT_DIR / f"{record_id}.png"
        if image_path.exists():
            image_rel_path = str(image_path.relative_to(ROOT))
            phash_value = hx_from_image(image_path)

    return {
        "id": record_id or "",
        "url": entry.get("url", ""),
        "label": "legit" if success else "legit_error",
        "bank": entry.get("bank", ""),
        "brand_state": derive_brand_state(entry.get("url", "")),
        "country": entry.get("country", ""),
        "lang": "",
        "ts_iso": entry.get("captured_at", ""),
        "viewport": f"{VIEWPORT['width']}x{VIEWPORT['height']}",
        "phash": phash_value,
        "source": "login_gallery_index",
        "http_status": status_code or "",
        "image_path": image_rel_path,
        "notes": error or "",
    }


def load_entries() -> List[Dict[str, Any]]:
    if not INPUT_JSON.exists():
        raise SystemExit(f"Missing input JSON at {INPUT_JSON}")
    with INPUT_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)
    entries = [item for item in data.get("results", []) if item.get("status") == "success" and item.get("url")]
    return entries


def write_manifest(rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "id",
        "url",
        "label",
        "bank",
        "brand_state",
        "country",
        "lang",
        "ts_iso",
        "viewport",
        "phash",
        "source",
        "http_status",
        "image_path",
        "notes",
    ]
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Manifest written to {MANIFEST_PATH} ({len(rows)} entries)")


def main() -> None:
    entries = load_entries()
    rows = asyncio.run(run(entries))
    write_manifest(rows)


if __name__ == "__main__":
    main()
