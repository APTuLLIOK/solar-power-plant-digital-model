import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_station_results(
        station_actual: pd.Series,
        station_model: pd.Series,
        save_path: str = None
        ):
    sns.set_theme(style="whitegrid", palette="muted")

    # 1. Факт vs Модель
    plt.figure(figsize=(15, 6))

    sns.lineplot(x=station_actual.index, y=station_actual.values,
                 label="Факт СЭС", linewidth=2.5, alpha=0.9)

    sns.lineplot(x=station_model.index, y=station_model.values,
                 label="Цифровая модель СЭС", linewidth=1.5, linestyle="--",
                 alpha=0.9)

    plt.title(
        "Солнечная электростанция: факт vs цифровая модель",
        fontsize=14,
        pad=15
        )
    plt.xlabel("Время", fontsize=12)
    plt.ylabel("AC мощность, кВт", fontsize=12)
    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        borderaxespad=0,
        frameon=True,
        shadow=True
        )
    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}_timeseries.png",
                    bbox_inches='tight', dpi=300)

    # 2. Остатки
    residuals = station_model - station_actual

    plt.figure(figsize=(15, 5))
    sns.lineplot(x=residuals.index, y=residuals.values,
                 label="Остаток (Модель - Факт)", color="indianred")

    plt.axhline(0, linestyle="--", color="black", alpha=0.7)
    plt.title("Невязка цифровой модели СЭС", fontsize=14, pad=15)
    plt.xlabel("Время", fontsize=12)
    plt.ylabel("Ошибка, кВт", fontsize=12)
    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        borderaxespad=0,
        frameon=True,
        shadow=True
        )
    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}_residuals.png", bbox_inches='tight', dpi=300)

    # 3. Распределение остатков
    plt.figure(figsize=(10, 5))

    sns.histplot(
        residuals.dropna(),
        bins=50,
        kde=True,
        color="steelblue",
        stat="count"
        )

    plt.title("Распределение остатков", fontsize=14, pad=15)
    plt.xlabel("Ошибка, кВт", fontsize=12)
    plt.ylabel("Количество", fontsize=12)
    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}_hist.png", bbox_inches='tight', dpi=300)

    print(f"Среднее значение остатка: {residuals.mean():.3f} кВт")
    print(f"Стандартное отклонение остатка: {residuals.std():.3f} кВт")

    plt.show()
