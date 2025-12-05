import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from services.infer.experiment_utils import load_dataset_with_aug, safe_filename

PLOTS_DIR = ROOT / "docs" / "Entrega_1___Fina" / "assets" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def generate_advanced_analysis(model_name: str, similarity_threshold: float = 0.85):
    print(f"\n--- Generando Análisis Avanzado para {model_name.upper()} (Umbral {similarity_threshold}) ---")

    data = load_dataset_with_aug(ROOT)
    model_results_dir = ROOT / "results" / f"detalle_{model_name}"
    
    all_scores = []
    all_true_labels = []
    hard_negatives = []
    
    scores_positive = []
    scores_negative = []

    queries = [r for r in data if r["label"] == "legit"]
    
    for query_data in queries:
        bank_name = query_data["bank"]
        fname = safe_filename(bank_name)
        csv_path = model_results_dir / f"comparativa_{fname}.csv"
        
        if not csv_path.exists():
            continue
        
        df_bank = pd.read_csv(csv_path)
        df_filtered = df_bank[~df_bank["is_self"]].copy()

        for _, row in df_filtered.iterrows():
            is_positive_class = (row["candidate_name"] == bank_name and 
                                 (row["candidate_type"] == "BANCO (REF)" or row["candidate_type"] == "BANCO (AUG)"))
            
            score = row["similarity_score"]
            all_scores.append(score)
            all_true_labels.append(1 if is_positive_class else 0)
            
            if is_positive_class:
                scores_positive.append(score)
            else:
                scores_negative.append(score)

            # Hard Negatives con el NUEVO UMBRAL
            if not is_positive_class and score > similarity_threshold:
                 hard_negatives.append({
                   "bank_target": bank_name,
                   "random_site": row["candidate_filename"],
                   "score": row["similarity_score"]
               })

    if not all_scores:
        return

    y_true = np.array(all_true_labels)
    y_scores = np.array(all_scores)

    # --- 1. Curva ROC ---
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'ROC - {model_name.upper()}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(PLOTS_DIR / f"roc_curve_{model_name}.png", bbox_inches='tight', dpi=300)
    plt.close()

    # --- 2. Matriz de Confusión (Normalizada) ---
    y_pred = (y_scores > similarity_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, normalize='true')

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues", 
                xticklabels=['Negativo', 'Positivo'],
                yticklabels=['Negativo', 'Positivo'])
    plt.title(f'Confusión ({model_name.upper()})\nUmbral > {similarity_threshold}')
    plt.savefig(PLOTS_DIR / f"confusion_matrix_{model_name}.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    # --- 3. Density Plot (KDE) - NUEVO ---
    # Muestra la superposición de densidades
    plt.figure(figsize=(8, 5))
    sns.kdeplot(scores_positive, fill=True, color="green", label="Positivos (Bancos)", alpha=0.3)
    sns.kdeplot(scores_negative, fill=True, color="red", label="Negativos (Random)", alpha=0.3)
    plt.axvline(x=similarity_threshold, color='black', linestyle='--', label=f'Umbral {similarity_threshold}')
    plt.title(f'Densidad de Probabilidad de Scores: {model_name.upper()}')
    plt.xlabel('Similitud Coseno')
    plt.xlim(0, 1.0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(PLOTS_DIR / f"kde_dist_{model_name}.png", bbox_inches='tight', dpi=300)
    plt.close()

def generate_comparative_kde():
    """Genera un KDE plot comparativo solo para SigLIP (el ganador) mostrando el corte limpio."""
    print("\n--- Generando KDE Comparativo para SigLIP ---")
    # Reutilizamos la lógica de carga para SigLIP solamente para simplificar el gráfico
    # Pero idealmente queremos ver los 3 modelos superpuestos? No, se ensucia mucho.
    # Mejor hacer un grid de 3 KDEs.
    pass # Ya lo estamos haciendo por modelo arriba

def generate_comparative_boxplot():
    print("\n--- Generando Comparativa Global de Distribuciones (Boxplot) ---")
    data_all = []
    
    models = ["clip", "siglip", "dinov2"]
    
    for model in models:
        data = load_dataset_with_aug(ROOT)
        model_results_dir = ROOT / "results" / f"detalle_{model}"
        queries = [r for r in data if r["label"] == "legit"]
        
        for query in queries:
            csv_path = model_results_dir / f"comparativa_{safe_filename(query['bank'])}.csv"
            if not csv_path.exists(): continue
            
            df = pd.read_csv(csv_path)
            df = df[~df["is_self"]]
            
            pos_mask = (df["candidate_name"] == query["bank"])
            neg_mask = (df["candidate_type"] == "RANDOM")
            
            pos_scores = df[pos_mask]["similarity_score"].tolist()
            neg_scores = df[neg_mask]["similarity_score"].tolist()
            
            for s in pos_scores:
                data_all.append({"Modelo": model.upper(), "Clase": "Positiva", "Similitud": s})
            for s in neg_scores:
                data_all.append({"Modelo": model.upper(), "Clase": "Negativa", "Similitud": s})
                
    df_all = pd.DataFrame(data_all)
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Modelo", y="Similitud", hue="Clase", data=df_all, palette="Set2")
    plt.axhline(y=0.85, color='r', linestyle='--', label='Umbral Operativo 0.85')
    plt.title("Separabilidad de Clases por Modelo (vs Umbral 0.85)")
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend(loc="lower right")
    plt.savefig(PLOTS_DIR / "comparativa_boxplots.png", bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    models_to_analyze = ["clip", "siglip", "dinov2"]
    # UMBRAL UNIFICADO 0.85
    COMMON_THRESHOLD = 0.85
    
    for model in models_to_analyze:
        generate_advanced_analysis(model, similarity_threshold=COMMON_THRESHOLD)
    
    generate_comparative_boxplot()
