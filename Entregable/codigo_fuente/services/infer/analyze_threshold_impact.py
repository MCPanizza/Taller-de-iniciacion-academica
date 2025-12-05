import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from services.infer.experiment_utils import load_dataset_with_aug, safe_filename

def analyze_siglip_thresholds():
    print("=== Análisis de Impacto de Umbral para SigLIP ===")
    model_name = "siglip"
    model_results_dir = ROOT / "results" / f"detalle_{model_name}"
    
    # Cargar dataset para saber cuáles son los queries
    data = load_dataset_with_aug(ROOT)
    queries = [r for r in data if r["label"] == "legit"]
    
    # Acumuladores
    scores_pos = []
    scores_neg = []
    
    for query in queries:
        bank_name = query["bank"]
        fname = safe_filename(bank_name)
        csv_path = model_results_dir / f"comparativa_{fname}.csv"
        
        if not csv_path.exists():
            continue
            
        df = pd.read_csv(csv_path)
        # Filtrar self-match
        df = df[~df["is_self"]]
        
        # Identificar positivos y negativos reales
        # Positivo: Es el mismo banco (original o aumentado)
        # Negativo: Es otro banco o random
        
        for _, row in df.iterrows():
            is_same_bank = (row["candidate_name"] == bank_name)
            is_legit_or_aug = (row["candidate_type"] in ["BANCO (REF)", "BANCO (AUG)"])
            
            is_positive_class = (is_same_bank and is_legit_or_aug)
            
            score = row["similarity_score"]
            
            if is_positive_class:
                scores_pos.append(score)
            else:
                # Solo nos interesan los negativos que son RANDOM para medir FPR operativo
                # (Confundir un banco con otro es malo, pero confundir random con banco es FP de alerta)
                # El usuario pidió enfoque en "falsos positivos", usualmente se refiere a alertas falsas de sitios random.
                # Pero para ser riguroso, cualquier cosa que no sea el banco es negativo.
                # Vamos a separar Random Negatives de Other Bank Negatives si es necesario, 
                # pero para el cálculo general de FP, usaremos todos los negativos.
                scores_neg.append({'score': score, 'type': row["candidate_type"], 'name': row["candidate_filename"]})

    scores_pos = np.array(scores_pos)
    scores_neg_vals = np.array([x['score'] for x in scores_neg])
    
    thresholds = [0.80, 0.85]
    
    print(f"\nTotal Muestras Positivas (Variantes): {len(scores_pos)}")
    print(f"Total Muestras Negativas (Ruido): {len(scores_neg_vals)}")
    
    for th in thresholds:
        # TP: Positivos detectados
        tp_count = np.sum(scores_pos > th)
        tpr = tp_count / len(scores_pos)
        
        # FP: Negativos detectados como positivos
        fp_count = np.sum(scores_neg_vals > th)
        fpr = fp_count / len(scores_neg_vals)
        
        # Ver cuáles son los FPs
        fps_details = [x for x in scores_neg if x['score'] > th]
        random_fps = [x for x in fps_details if x['type'] == 'RANDOM']
        
        print(f"\n--- Umbral > {th} ---")
        print(f"True Positives (Recuperados): {tp_count} / {len(scores_pos)} ({tpr:.2%})")
        print(f"False Positives (Total): {fp_count} / {len(scores_neg_vals)} ({fpr:.4%})")
        print(f"False Positives (Solo Random): {len(random_fps)}")
        
        if len(random_fps) > 0:
            print("Ejemplos de FP Random:")
            for fp in random_fps[:5]:
                print(f"  - {fp['name']} (Score: {fp['score']:.4f})")

if __name__ == "__main__":
    analyze_siglip_thresholds()

