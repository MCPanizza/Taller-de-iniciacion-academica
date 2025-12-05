import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "docs" / "Entrega_1___Fina" / "assets" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def load_all_detail_csvs(model_name):
    """Carga todos los CSVs de detalle para un modelo."""
    detail_dir = RESULTS_DIR / f"detalle_{model_name}"
    all_files = list(detail_dir.glob("*.csv"))
    
    dfs = []
    for f in all_files:
        df = pd.read_csv(f)
        dfs.append(df)
    
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

def plot_similarity_distribution(model_name="siglip", threshold=0.80):
    print(f"Generando distribución para {model_name}...")
    df = load_all_detail_csvs(model_name)
    
    if df.empty:
        print(f"No se encontraron datos para {model_name}")
        return

    # Filtrar:
    # - Positivos: True Positives (Originales y Aumentados del mismo banco)
    # - Negativos: Random / Otros Bancos (Falsos positivos potenciales)
    
    # Identificar TPs: is_true_positive es True, o es self-match (que también es TP trivial)
    # Para el gráfico de "Identidad", usamos Self y Augmented.
    # Pero para ser estrictos con el desafío, usemos solo Augmented vs Random.
    
    # Scores de Verdaderos Positivos (Variantes aumentadas)
    tp_scores = df[df["is_true_positive"] == True]["similarity_score"]
    
    # Scores de Negativos (Cualquier cosa que NO sea el banco target)
    # Excluimos self y true positives
    fp_scores = df[
        (df["is_self"] == False) & 
        (df["is_true_positive"] == False)
    ]["similarity_score"]
    
    plt.figure(figsize=(10, 6))
    
    # Histograma Negativos (Randoms)
    sns.histplot(fp_scores, color="red", label="Sitios Random / Otros (Ruido)", stat="density", bins=50, alpha=0.5)
    
    # Histograma Positivos (Bancos Correctos)
    sns.histplot(tp_scores, color="green", label="Bancos Correctos (Variantes)", stat="density", bins=50, alpha=0.6)
    
    # Línea de Umbral
    plt.axvline(threshold, color='blue', linestyle='--', linewidth=2, label=f'Umbral Decisión ({threshold})')
    
    plt.title(f"Separación de Clases - Modelo {model_name.upper()}", fontsize=14)
    plt.xlabel("Puntaje de Similitud Coseno", fontsize=12)
    plt.ylabel("Densidad", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1.0)
    
    out_path = PLOTS_DIR / f"distribucion_{model_name}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado: {out_path}")

def plot_model_comparison():
    print("Generando comparativa de modelos...")
    
    # Cargar datos de TPR
    data = []
    models = ["clip", "siglip", "dinov2"]
    
    for m in models:
        tpr_path = RESULTS_DIR / f"tpr_analysis_{m}.csv"
        if tpr_path.exists():
            df = pd.read_csv(tpr_path)
            tpr = df["top_match_is_aug"].mean() * 100
            conf = df["top_match_score"].mean()
            data.append({"Modelo": m.upper(), "Métrica": "TPR @ Top-1 (%)", "Valor": tpr})
            data.append({"Modelo": m.upper(), "Métrica": "Confianza Promedio", "Valor": conf * 100})
            
    if not data:
        return

    df_plot = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_plot, x="Modelo", y="Valor", hue="Métrica", palette="viridis")
    
    plt.title("Comparativa de Robustez entre Modelos", fontsize=14)
    plt.ylim(80, 105) # Zoom en la parte superior porque todos son buenos
    plt.ylabel("Puntaje / Porcentaje", fontsize=12)
    plt.grid(True, axis='y', alpha=0.3)
    
    # Etiquetas de valor
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=3)
        
    out_path = PLOTS_DIR / "comparativa_modelos.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado: {out_path}")

if __name__ == "__main__":
    # Configurar estilo
    sns.set_theme(style="whitegrid")
    
    plot_similarity_distribution("siglip", threshold=0.80)
    plot_model_comparison()

