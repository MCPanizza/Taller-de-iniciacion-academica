import sys
from pathlib import Path
import csv
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score, average_precision_score
from datetime import datetime
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from phishvision.vision.backends import VisionEncoder

def load_dataset_with_aug(root_path):
    manifest_path = root_path / "datasets" / "mini" / "manifest.csv"
    print(f"Cargando dataset desde {manifest_path}...")
    
    rows = []
    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rel_path = Path(r["image_path"])
            
            # Construcción robusta
            if r["label"] == "legit":
                full_path = root_path / "datasets" / "mini" / "screenshots_legit" / rel_path.name
            elif r["label"] == "legit_aug":
                full_path = root_path / "datasets" / "mini" / "screenshots_aug" / rel_path.name
            else:
                full_path = root_path / "datasets" / "mini" / "screenshots_random" / rel_path.name
            
            if full_path.exists():
                r["full_path"] = str(full_path)
                r["filename"] = rel_path.name
                rows.append(r)
    return rows

def safe_filename(name):
    return "".join([c if c.isalnum() or c in ('-','_') else '_' for c in name])

def run_full_experiment(model_name, root_path):
    print(f"\n==================================================")
    print(f"INICIANDO EXPERIMENTO COMPLETO: {model_name.upper()}")
    print(f"==================================================")
    
    # 1. Cargar Datos
    data = load_dataset_with_aug(root_path)
    
    # Definir Conjuntos
    # Queries: Solo Bancos Originales (queremos ver qué encuentran)
    queries = [r for r in data if r["label"] == "legit"]
    
    # Universo: Todo (Originales + Aumentados + Random)
    universe = data
    
    print(f"Queries (Bancos): {len(queries)}")
    print(f"Universo de Búsqueda: {len(universe)}")
    
    # 2. Cargar Modelo
    try:
        encoder = VisionEncoder(model_name)
    except Exception as e:
        print(f"Error cargando modelo {model_name}: {e}")
        return
    
    # 3. Generar Embeddings del Universo
    print("Generando embeddings del universo completo...")
    universe_paths = [r["full_path"] for r in universe]
    universe_embeds = encoder.encode_image(universe_paths)
    
    # Mapear índices para acceso rápido
    # Necesitamos saber cuáles índices del universo corresponden a nuestros queries
    query_indices_in_universe = [i for i, r in enumerate(universe) if r["label"] == "legit"]
    
    # 4. Métricas Globales (AUROC/AUPRC)
    # Para esto, consideramos:
    # Clase Positiva (1): legit y legit_aug
    # Clase Negativa (0): random
    print("Calculando métricas globales...")
    
    # Separar embeddings por clase para cálculo vectorial eficiente
    pos_indices = [i for i, r in enumerate(universe) if r["label"] in ["legit", "legit_aug"]]
    neg_indices = [i for i, r in enumerate(universe) if "random" in r["label"]]
    
    pos_embeds = universe_embeds[pos_indices]
    neg_embeds = universe_embeds[neg_indices]
    
    # Similitud Positiva (Intra-clase, ignorando auto-match)
    sim_pos_matrix = cosine_similarity(pos_embeds, pos_embeds)
    np.fill_diagonal(sim_pos_matrix, -1)
    sim_pos_scores = sim_pos_matrix.max(axis=1)
    
    # Similitud Negativa (Random vs Bancos)
    sim_neg_scores = cosine_similarity(neg_embeds, pos_embeds).max(axis=1)
    
    y_true = [1] * len(sim_pos_scores) + [0] * len(sim_neg_scores)
    y_scores = np.concatenate([sim_pos_scores, sim_neg_scores])
    
    auroc = roc_auc_score(y_true, y_scores)
    auprc = average_precision_score(y_true, y_scores)
    
    print(f"--> AUROC Global: {auroc:.4f}")
    print(f"--> AUPRC Global: {auprc:.4f}")
    
    # Guardar métricas globales
    metrics_csv = root_path / "results" / "benchmark_metrics.csv"
    new_row = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "auroc": auroc,
        "auprc": auprc,
        "n_queries": len(queries),
        "n_universe": len(universe)
    }
    
    # Append seguro
    header = not metrics_csv.exists()
    pd.DataFrame([new_row]).to_csv(metrics_csv, mode='a', header=header, index=False)
    
    # 5. Generar Reportes Detallados por Banco (CSV)
    out_dir = root_path / "results" / f"detalle_{model_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generando reportes detallados en {out_dir}...")
    
    tpr_stats = [] # Para calcular tasa de éxito de recuperación de aumentados
    
    for q_idx in tqdm(query_indices_in_universe, desc="Procesando Bancos"):
        query_vec = universe_embeds[q_idx].reshape(1, -1)
        query_data = universe[q_idx]
        bank_name = query_data['bank']
        
        sims = cosine_similarity(query_vec, universe_embeds)[0]
        
        results = []
        found_aug = False
        
        for i, score in enumerate(sims):
            candidate = universe[i]
            
            c_type = "RANDOM"
            if candidate['label'] == "legit":
                c_type = "BANCO (REF)"
            elif candidate['label'] == "legit_aug":
                c_type = "BANCO (AUG)"
            
            # Lógica de Decisión con Umbral > 80%
            THRESHOLD = 0.80
            is_match = score > THRESHOLD
            
            is_true_positive = False
            # Es True Positive si:
            # 1. Supera el umbral
            # 2. El banco candidato es el MISMO que el banco target
            # 3. El candidato es legítimo (original o aumentado)
            if is_match and candidate['bank'] == bank_name and candidate['label'] in ["legit", "legit_aug"]:
                is_true_positive = True
            
            # Es False Positive si:
            # 1. Supera el umbral
            # 2. NO es el mismo banco (puede ser otro banco o random)
            is_false_positive = False
            if is_match and not is_true_positive:
                is_false_positive = True
            
            # Identidad (Self-match siempre es 1.0, lo marcamos aparte)
            is_self = (i == q_idx)
            
            results.append({
                "target_bank": bank_name,
                "candidate_filename": candidate['filename'],
                "candidate_type": c_type,
                "candidate_name": candidate.get('bank', 'Random'),
                "similarity_score": score,
                "is_match_over_80": is_match,
                "is_true_positive": is_true_positive,
                "is_false_positive": is_false_positive,
                "is_self": is_self
            })
        
        # Guardar CSV individual
        df = pd.DataFrame(results).sort_values("similarity_score", ascending=False)
        fname = safe_filename(bank_name)
        df.to_csv(out_dir / f"comparativa_{fname}.csv", index=False)
        
        # Registrar si encontró su aumentado en el Top 1 (después de sí mismo)
        # Filtramos self
        df_noself = df[~df["is_self"]]
        if not df_noself.empty:
            top_match = df_noself.iloc[0]
            tpr_stats.append({
                "bank": bank_name,
                "top_match_is_aug": top_match["is_true_positive"],
                "top_match_score": top_match["similarity_score"]
            })
            
    # Guardar resumen de TPR
    tpr_df = pd.DataFrame(tpr_stats)
    tpr_path = root_path / "results" / f"tpr_analysis_{model_name}.csv"
    tpr_df.to_csv(tpr_path, index=False)
    
    print(f"Experimento {model_name} finalizado.")
    print(f"Resumen TPR guardado en {tpr_path}")

