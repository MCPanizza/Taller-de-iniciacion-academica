import csv
from pathlib import Path
import imagehash
from PIL import Image
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[3]
LEGIT_MANIFEST = ROOT / "datasets" / "mini" / "legit_manifest.csv"
RANDOM_MANIFEST = ROOT / "datasets" / "mini" / "random_manifest.csv"
OUTPUT_MANIFEST = ROOT / "datasets" / "mini" / "manifest.csv"

HAMMING_THRESHOLD = 8  # Umbral estándar para similaridad pHash

def load_manifest(path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Aceptamos "legit" (bancos) y "random_legit" (control)
            if r.get("label") in ["legit", "random_legit"] and r.get("phash"):
                rows.append(r)
    return rows

def deduplicate(rows):
    unique_hashes = {}
    deduped = []
    duplicates_count = 0
    
    print(f"Procesando {len(rows)} entradas...")
    
    for row in rows:
        phash_str = row["phash"]
        try:
            # Convertir hex string a objeto ImageHash
            current_hash = imagehash.hex_to_hash(phash_str)
        except Exception:
            continue
            
        is_duplicate = False
        
        # Comparar contra hashes ya vistos
        for stored_hash_str, stored_row in unique_hashes.items():
            stored_hash = imagehash.hex_to_hash(stored_hash_str)
            if current_hash - stored_hash <= HAMMING_THRESHOLD:
                is_duplicate = True
                duplicates_count += 1
                break
        
        if not is_duplicate:
            unique_hashes[phash_str] = row
            deduped.append(row)
            
    print(f"Eliminados {duplicates_count} duplicados visuales (Hamming <= {HAMMING_THRESHOLD})")
    return deduped

def main():
    print("Cargando manifiestos...")
    legit = load_manifest(LEGIT_MANIFEST)
    random_legit = load_manifest(RANDOM_MANIFEST)
    
    print(f"Bancos Legítimos: {len(legit)}")
    print(f"Random Control:   {len(random_legit)}")
    
    # Combinar y deduplicar
    all_rows = legit + random_legit
    
    # Deduplicación
    final_dataset = deduplicate(all_rows)
    
    # Estadísticas finales
    legit_count = sum(1 for r in final_dataset if r["label"] == "legit")
    random_count = sum(1 for r in final_dataset if r["label"] == "random_legit")
    
    print("-" * 40)
    print("ESTADÍSTICAS FINALES DEL DATASET (Modo Seguro)")
    print("-" * 40)
    print(f"Bancos (Target): {legit_count}")
    print(f"Random (Control): {random_count}")
    print(f"Total:            {len(final_dataset)}")
    print("-" * 40)
    
    # Guardar manifest unificado
    if final_dataset:
        fieldnames = final_dataset[0].keys()
        with OUTPUT_MANIFEST.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(final_dataset)
        print(f"Manifest unificado guardado en: {OUTPUT_MANIFEST}")

if __name__ == "__main__":
    main()

