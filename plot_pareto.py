# plot_pareto.py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import sys
import os

def plot_pareto_3d(csv_path: str, out_dir: str = None):
    df = pd.read_csv(csv_path)

    # поддержка и 2-целевого и 3-целевого CSV
    has_passengers = "total_served" in df.columns

    if not has_passengers:
        print("Колонка total_served не найдена — строим 2D график")
        _plot_2d(df, csv_path, out_dir)
        return

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    x = df["total_tram_km"].values
    y = df["schedule_mae_min"].values
    z = df["total_served"].values

    # цвет = суммарный парк
    total_trams = df[["n_20", "n_48", "n_55"]].sum(axis=1).values
    norm = plt.Normalize(total_trams.min(), total_trams.max())
    colors = cm.viridis(norm(total_trams))

    sc = ax.scatter(x, y, z, c=total_trams, cmap="viridis",
                    s=80, edgecolors="k", linewidths=0.4, alpha=0.9)

    # аннотируем крайние точки
    extremes = {
        "мин km":  df["total_tram_km"].idxmin(),
        "мин mae": df["schedule_mae_min"].idxmin(),
        "макс пасс": df["total_served"].idxmax(),
    }
    for label, idx in extremes.items():
        ax.text(
            df.loc[idx, "total_tram_km"],
            df.loc[idx, "schedule_mae_min"],
            df.loc[idx, "total_served"],
            f'  {label}\n  [{int(df.loc[idx,"n_20"])},{int(df.loc[idx,"n_48"])},{int(df.loc[idx,"n_55"])}]',
            fontsize=7, color="darkred"
        )

    cbar = fig.colorbar(sc, ax=ax, pad=0.1, shrink=0.6)
    cbar.set_label("Суммарный парк (трамваев)", fontsize=9)

    ax.set_xlabel("total_tram_km", fontsize=9, labelpad=10)
    ax.set_ylabel("schedule_mae (мин)\n(точность расписания)", fontsize=9, labelpad=10)
    ax.set_zlabel("Пассажиры\n(обслужено)", fontsize=9, labelpad=10)
    ax.set_title("Pareto-фронт NSGA-II\nМаршруты 20, 48, 55", fontsize=12, pad=15)

    ax.view_init(elev=20, azim=45)  # угол обзора — меняй по вкусу
    plt.tight_layout()

    # сохраняем
    out_dir = out_dir or os.path.dirname(csv_path)
    out_path = os.path.join(out_dir, "pareto_front_3d.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"График сохранён: {out_path}")
    plt.show()


def _plot_2d(df, csv_path, out_dir):
    """Fallback — 2D если нет колонки пассажиров"""
    fig, ax = plt.subplots(figsize=(9, 6))
    total_trams = df[["n_20", "n_48", "n_55"]].sum(axis=1).values
    sc = ax.scatter(
        df["total_tram_km"], df["schedule_mae_min"],
        c=total_trams, cmap="viridis", s=80, edgecolors="k", linewidths=0.4
    )
    plt.colorbar(sc, ax=ax, label="Суммарный парк (трамваев)")
    ax.set_xlabel("total_tram_km (затраты)")
    ax.set_ylabel("schedule_mae (мин)")
    ax.set_title("Pareto-фронт NSGA-II — маршруты 20, 48, 55")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_dir = out_dir or os.path.dirname(csv_path)
    out_path = os.path.join(out_dir, "pareto_front_2d.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"График сохранён: {out_path}")
    plt.show()


if __name__ == "__main__":
    # использование: python plot_pareto.py outputs/nsga2_test/pareto_front.csv
    if len(sys.argv) < 2:
        print("Использование: python plot_pareto.py <путь до pareto_front.csv>")
        sys.exit(1)

    plot_pareto_3d(csv_path=sys.argv[1])
