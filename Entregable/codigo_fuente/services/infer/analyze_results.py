import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

def analyze_experiments():
    print("=== ANÁLISIS ESTADÍSTICO FINAL ===")
    
    results_dir = ROOT / "results"
    
    # 1. Análisis Global (AUROC/AUPRC)
    metrics_path = results_dir / "benchmark_metrics.csv"
    if metrics_path.exists():
        df_metrics = pd.read_csv(metrics_path)
        print("\n--- Métricas Globales por Modelo ---")
        # Tomar la última ejecución de cada modelo
        latest_metrics = df_metrics.sort_values("timestamp").groupby("model").tail(1)
        print(latest_metrics[["model", "auroc", "auprc"]])
        
        # Guardar tabla para LaTeX
        latex_table = latest_metrics[["model", "auroc", "auprc"]].to_latex(index=False, float_format="%.4f")
        with open(results_dir / "final_metrics_table.tex", "w") as f:
            f.write(latex_table)
    else:
        print("No se encontró benchmark_metrics.csv")

    # 2. Análisis de Robustez (TPR en Data Augmentation)
    models = ["clip", "siglip", "dinov2"]
    tpr_summary = []
    
    for model in models:
        tpr_path = results_dir / f"tpr_analysis_{model}.csv"
        if tpr_path.exists():
            df_tpr = pd.read_csv(tpr_path)
            
            # TPR@Top1: % de veces que la imagen aumentada fue el primer match
            tpr_top1 = df_tpr["top_match_is_aug"].mean()
            avg_score = df_tpr["top_match_score"].mean()
            
            tpr_summary.append({
                "model": model,
                "TPR@Top1": tpr_top1,
                "Avg_Confidence": avg_score
            })
            
    if tpr_summary:
        df_summary = pd.DataFrame(tpr_summary)
        print("\n--- Robustez (Data Augmentation) ---")
        print(df_summary)
        
        # Guardar tabla TPR
        latex_tpr = df_summary.to_latex(index=False, float_format="%.4f")
        with open(results_dir / "final_tpr_table.tex", "w") as f:
            f.write(latex_tpr)
            
    print("\nAnálisis completo. Tablas LaTeX generadas en results/")

if __name__ == "__main__":
    analyze_experiments()
