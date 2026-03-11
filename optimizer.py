# optimizer.py
from __future__ import annotations

import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from simulation.multi_route import MultiRouteSimulation

logging.basicConfig(level=logging.ERROR)  # глушим SimPy-логи во время оптимизации

# ─── конфиги маршрутов ────────────────────────────────────────────────────────
ROUTE_PAIRS = {
    "20": ("configs/route_20_fwd_config.json", "configs/route_20_bwd_config.json"),
    "48": ("configs/route_48_fwd_config.json", "configs/route_48_bwd_config.json"),
    "55": ("configs/route_55_fwd_config.json", "configs/route_55_bwd_config.json"),
}

N_ROUTES = len(ROUTE_PAIRS)   # 3
N_MAX    = 45                  # бюджет парка — подбери под реальность


# ─── задача оптимизации ───────────────────────────────────────────────────────

class TramFleetProblem(ElementwiseProblem):
    """
    Переменные:   x = [n_20, n_48, n_55]  — целые, диапазон [5, 30]
    Цели:         F = [total_tram_km, schedule_mae]  — минимизируем обе
    Ограничения:  G = [sum(x) - N_MAX]  ≤ 0
    """

    def __init__(self, n_max: int = N_MAX):
        super().__init__(
            n_var=N_ROUTES,
            n_obj=3,
            n_ieq_constr=1,
            xl=np.full(N_ROUTES, 5),
            xu=np.full(N_ROUTES, 30),
            vtype=int,
        )
        self.n_max = n_max

    def _evaluate(self, x, out, *args, **kwargs):
        tram_counts = [int(v) for v in x]

        # silent=True — не создаём папки на диске
        sim = MultiRouteSimulation.from_params(
            ROUTE_PAIRS,
            tram_counts=tram_counts,
            run_dir=None,   # <— папка не создаётся
        )
        sim.run(plot_graphs=False, save_logs=False)

        _, total_km, schedule_mae, total_served = sim.get_objectives()

        out["F"] = np.array([-total_km, schedule_mae, -total_served], dtype=float)
        out["G"] = np.array([sum(tram_counts) - self.n_max], dtype=float)


# ─── запуск NSGA-II ───────────────────────────────────────────────────────────

def run_nsga2(
    n_max:    int = N_MAX,
    pop_size: int = 40,
    n_gen:    int = 50,
    seed:     int = 42,
    out_dir:  str = "outputs/nsga2",
) -> tuple:
    os.makedirs(out_dir, exist_ok=True)

    problem = TramFleetProblem(n_max=n_max)

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()),
        mutation=PM(eta=20, vtype=float, repair=RoundingRepair()),
        eliminate_duplicates=True,
    )

    termination = get_termination("n_gen", n_gen)

    print(f"Запуск NSGA-II: pop={pop_size}, gen={n_gen}, n_max={n_max}")
    print(f"Всего evaluations: ~{pop_size * n_gen}")

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=seed,
        verbose=True,
        save_history=False,
    )

    # ─── сохраняем Pareto-фронт ───────────────────────────────────────────────
    _save_results(res, out_dir)

    return res


def _save_results(res, out_dir: str):
    X = res.X   # переменные: [n_20, n_48, n_55]
    F = res.F   # цели:       [total_km, schedule_mae]

    # CSV
    df = pd.DataFrame(
        np.hstack([X, F]),
        columns=["n_20", "n_48", "n_55", "total_tram_km_neg", "schedule_mae_min", "total_served_neg"],
    )
    df["total_tram_km"] = -df["total_tram_km_neg"]
    df["total_served"]  = -df["total_served_neg"]
    df = df.drop(columns=["total_tram_km_neg", "total_served_neg"])   # возвращаем знак обратно
    df["total_trams"] = df[["n_20", "n_48", "n_55"]].sum(axis=1)

    df = df.sort_values("total_tram_km")
    csv_path = os.path.join(out_dir, "pareto_front.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nPareto-фронт сохранён: {csv_path}")
    print(df.to_string(index=False))

    # График
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        F[:, 0], F[:, 1],
        c=X.sum(axis=1),          # цвет = суммарный парк
        cmap="viridis", s=80, edgecolors="k", linewidths=0.5,
    )
    plt.colorbar(sc, ax=ax, label="Суммарный парк (трамваев)")
    ax.set_xlabel("total_tram_km")
    ax.set_ylabel("schedule_mae (мин)")
    ax.set_title("Pareto-фронт NSGA-II: маршруты 20, 48, 55")
    ax.grid(True, alpha=0.3)

    # аннотируем крайние точки
    for idx in [0, len(F) - 1]:
        ax.annotate(
            f"[{int(X[idx,0])},{int(X[idx,1])},{int(X[idx,2])}]",
            (F[idx, 0], F[idx, 1]),
            textcoords="offset points", xytext=(6, 6), fontsize=8,
        )

    plot_path = os.path.join(out_dir, "pareto_front.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"График сохранён: {plot_path}")


# ─── точка входа ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_nsga2(n_max=45, pop_size=40, n_gen=50)
