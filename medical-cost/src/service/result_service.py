from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src.interfaces.abstract import RegressionModel

def save_results(results: pd.DataFrame, output_path: Path) -> None:
    """Guarda la tabla comparativa de los modelos."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    results.to_csv(output_path, index=False)

def print_comparison(results: pd.DataFrame) -> None:
    """Muestra la tabla comparativa ordenada del mejor al peor modelo."""

    table = results.sort_values("r2_test", ascending=False).copy()

    print("\n" + "=" * 50)
    print("COMPARACIÓN DE MODELOS (conjunto de prueba)")
    print("=" * 50)

    print(
        table.to_string(
            index=False,
            formatters={
                "r2_train": "{:.4f}".format,
                "r2_test": "{:.4f}".format,
                "r2_cv_mean": "{:.4f}".format,
                "r2_cv_std": "{:.4f}".format,
                "mae_test": "${:,.0f}".format,
                "rmse_test": "${:,.0f}".format,
                "mape_test": "{:.1f}%".format,
            },
        )
    )

    best = table.iloc[0]

    print(f"\nMejor modelo: {best['model']} (R2 = {best['r2_test']:.4f}, MAE = ${best['mae_test']:,.0f})")

def save_comparison_figures(
    results: pd.DataFrame,
    models: list[RegressionModel],
    x_test: pd.DataFrame,
    y_test: pd.Series,
    degree_results: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Genera las gráficas de comparación que se usan en el informe."""

    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Métricas por modelo.
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (column, title) in zip(
        axes,
        [
            ("r2_test", "R² (más alto es mejor)"),
            ("mae_test", "MAE en USD (más bajo es mejor)"),
            ("rmse_test", "RMSE en USD (más bajo es mejor)"),
        ],
    ):
        sns.barplot(data=results, x="model", y=column, ax=ax, color="#4C72B0")
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=15)

        for container in ax.containers:
            labels = [f"{v:.3f}" if column == "r2_test" else f"{v:,.0f}" for v in container.datavalues]
            ax.bar_label(container, labels=labels)

    fig.tight_layout()
    fig.savefig(output_dir / "05_comparacion_metricas.png", dpi=120)
    plt.close(fig)

    # Valor real vs. predicho: un modelo perfecto deja todos los puntos sobre la diagonal.
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), sharey=True)
    limit = max(y_test.max(), 1) * 1.05

    for ax, model in zip(axes, models):
        predictions = model.predict(x_test)

        ax.scatter(y_test, predictions, alpha=0.5, s=18)
        ax.plot([0, limit], [0, limit], color="red", linestyle="--", linewidth=1)
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_title(model.name)
        ax.set_xlabel("Costo real (USD)")
        ax.set_ylabel("Costo predicho (USD)")

    fig.tight_layout()
    fig.savefig(output_dir / "06_real_vs_predicho.png", dpi=120)
    plt.close(fig)

    # Selección del grado del polinomio.
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(degree_results["degree"], degree_results["r2_train"], marker="o", label="Entrenamiento")
    ax.errorbar(
        degree_results["degree"],
        degree_results["r2_cv_mean"],
        yerr=degree_results["r2_cv_std"],
        marker="o",
        capsize=4,
        label="Validación cruzada (5 particiones)",
    )
    ax.set_xticks(degree_results["degree"])
    ax.set_title("Regresión polinómica: R² según el grado")
    ax.set_xlabel("Grado del polinomio")
    ax.set_ylabel("R²")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "07_seleccion_grado_polinomio.png", dpi=120)
    plt.close(fig)
