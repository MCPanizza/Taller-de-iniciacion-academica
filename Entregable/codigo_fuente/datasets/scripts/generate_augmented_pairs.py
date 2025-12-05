import sys
from pathlib import Path
import csv
import uuid
from datetime import datetime
from PIL import Image, ImageEnhance, ImageFilter
import random
import pandas as pd
import shutil

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

def clean_previous_aug(aug_dir):
    """Limpia aumentos previos para no acumular basura."""
    if aug_dir.exists():
        shutil.rmtree(aug_dir)
    aug_dir.mkdir(parents=True, exist_ok=True)

def apply_random_transform(img):
    """Aplica una combinación aleatoria de transformaciones."""
    w, h = img.size
    
    # 1. Crop Aleatorio (Simula viewport o scroll)
    # Mantenemos entre el 80% y 95% de la imagen
    crop_factor = random.uniform(0.80, 0.95)
    new_w = int(w * crop_factor)
    new_h = int(h * crop_factor)
    
    # Desplazamiento aleatorio del crop (no siempre centrado)
    max_x = w - new_w
    max_y = h - new_h
    left = random.randint(0, max_x)
    top = random.randint(0, max_y // 2) # Preferimos mantener el header visible
    
    img = img.crop((left, top, left + new_w, top + new_h))
    
    # 2. Resize (Simula escalado)
    img = img.resize((800, 600), Image.Resampling.LANCZOS)
    
    # 3. Variaciones de Color/Brillo (Simula monitores/renderizado)
    if random.random() > 0.5:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))
        
    if random.random() > 0.5:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))
        
    # 4. Blur leve o Ruido (Simula compresión o mala calidad)
    if random.random() > 0.7:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.0)))
        
    return img

def generate_augmented_dataset(n_variations=10):
    print(f"=== Generando {n_variations} Variaciones por Banco para Validación Robusta ===")
    
    MANIFEST_PATH = ROOT / "datasets" / "mini" / "manifest.csv"
    AUG_DIR = ROOT / "datasets" / "mini" / "screenshots_aug"
    
    # Limpiar directorio anterior
    clean_previous_aug(AUG_DIR)
    
    # Cargar manifest actual y LIMPIARLO de aumentos viejos
    df = pd.read_csv(MANIFEST_PATH)
    
    # Quedarnos solo con legit y random originales
    df_clean = df[df["label"] != "legit_aug"].copy()
    
    # Filtrar solo legitimos originales para procesar
    legit_rows = df_clean[df_clean["label"] == "legit"]
    
    new_rows = []
    
    print(f"Procesando {len(legit_rows)} bancos originales...")
    
    for _, row in legit_rows.iterrows():
        original_path = ROOT / row["image_path"]
        
        if not original_path.exists():
            continue
            
        try:
            # Cargar imagen original una vez
            base_img = Image.open(original_path).convert("RGB")
            
            for i in range(n_variations):
                # Aplicar transformación
                aug_img = apply_random_transform(base_img.copy())
                
                # Guardar
                timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
                aug_filename = f"legit_aug_{row['bank'].replace(' ', '-').lower()}_v{i}_{timestamp}.png"
                output_path = AUG_DIR / aug_filename
                aug_img.save(output_path, quality=85) # Simular compresión JPEG también
                
                # Metadata
                new_row = row.copy()
                new_row["id"] = f"aug_{uuid.uuid4().hex[:8]}"
                new_row["label"] = "legit_aug"
                new_row["image_path"] = str(output_path.relative_to(ROOT))
                new_row["notes"] = f"Variation {i+1}/{n_variations} (Crop/Color/Blur)"
                new_row["ts_iso"] = datetime.now().isoformat()
                
                new_rows.append(new_row)
                
        except Exception as e:
            print(f"Error procesando {row['bank']}: {e}")
            
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        final_df = pd.concat([df_clean, new_df], ignore_index=True)
        
        # Guardar
        final_df.to_csv(MANIFEST_PATH, index=False)
        
        print(f"\n¡Éxito! Se generaron {len(new_rows)} imágenes aumentadas.")
        print(f"Total Dataset: {len(final_df)} (Legit: {len(legit_rows)}, Aug: {len(new_rows)})")
    else:
        print("No se generaron nuevas imágenes.")

if __name__ == "__main__":
    generate_augmented_dataset(n_variations=10)
