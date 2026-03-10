"""
MultiRouteSimulation — оркестратор нескольких маршрутов в едином env.

Использование:
    sim = MultiRouteSimulation(["configs/r20.json", "configs/r48.json"])
    sim.run()

Для NSGA-II:
    sim = MultiRouteSimulation.from_params(base_configs, tram_counts, intervals)
    sim.run(plot_graphs=False, save_logs=False)
    cost = sim.get_objectives()
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import simpy

from models.stop import Stop
from models.route import Route, RouteConfig
from models.tram import Tram
from logger import TramLogger
from visualization import TramVisualization

log = logging.getLogger(__name__)

OUTPUT_DIR = "outputs"


class MultiRouteSimulation:

    def __init__(self, config_files: List[str], run_dir: Optional[str] = None):
        self.env = simpy.Environment()
        self.shared_stops: Dict[int, Stop] = {}
        self.routes: List[Route] = []
        self.run_dir = run_dir or self._create_run_directory()

        tram_id_offset = 0
        for cf in config_files:
            cfg = RouteConfig.from_json(cf)
            self._register_stops(cfg)
            route = Route(cfg, self.env, self.shared_stops, tram_id_offset=tram_id_offset)
            self.routes.append(route)
            tram_id_offset += cfg.tram_count   # глобально уникальные ID трамваев

        log.info(
            f"MultiRouteSimulation: {len(self.routes)} маршрут(а/ов), "
            f"{len(self.shared_stops)} уникальных остановок"
        )

    # ── Фабричный метод для оптимизатора ──────────────────────────────────────

    @classmethod
    def from_params(
        cls,
        config_files: List[str],
        tram_counts: List[int],
        intervals_override: Optional[List[List[Tuple[int, int]]]] = None,
        run_dir: Optional[str] = None,
    ) -> "MultiRouteSimulation":
        """
        Создаёт симуляцию с переопределёнными параметрами (для NSGA-II).

        tram_counts        — [count_r1, count_r2, ...]
        intervals_override — [[(start_h, interval_min), ...], ...] или None
        """
        import copy, tempfile, json

        patched_files = []
        tmp_dir = tempfile.mkdtemp()

        for i, cf in enumerate(config_files):
            with open(cf, "r", encoding="utf-8") as f:
                cfg_data = json.load(f)

            cfg_data["tram_count"] = tram_counts[i]
            if intervals_override and i < len(intervals_override):
                cfg_data["bus_interval"] = intervals_override[i]

            tmp_path = os.path.join(tmp_dir, f"route_{i}.json")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(cfg_data, f, ensure_ascii=False)
            patched_files.append(tmp_path)

        return cls(patched_files, run_dir=run_dir)

    # ── Регистрация остановок ─────────────────────────────────────────────────

    def _register_stops(self, cfg: RouteConfig):
        """Идемпотентно: создаёт остановку только если её ещё нет в реестре."""
        for stop_id in cfg.stop_ids:
            if stop_id not in self.shared_stops:
                self.shared_stops[stop_id] = Stop(stop_id, self.env)

    # ── Вспомогательные ───────────────────────────────────────────────────────

    def _create_run_directory(self) -> str:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = os.path.join(OUTPUT_DIR, f"run_{ts}")
        os.makedirs(os.path.join(run_dir, "logs"),  exist_ok=True)
        os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
        log.info(f"Результаты: {run_dir}/")
        return run_dir

    def _max_hours(self) -> int:
        return max(r.config.simulation_hours for r in self.routes)

    def _all_trams(self) -> Dict[int, Tram]:
        result = {}
        for route in self.routes:
            result.update(route.trams)
        return result

    # ── Запуск ────────────────────────────────────────────────────────────────

    def run(self, plot_graphs: bool = True, save_logs: bool = True):
        for route in self.routes:
            route.start()

        self.env.run(until=self._max_hours() * 60)
        self._print_stats()

        if save_logs:
            logs_dir = os.path.join(self.run_dir, "logs")
            for route in self.routes:
                tl = TramLogger(output_dir=os.path.join(logs_dir, route.config.route_id))
                tl.save_all_trams(route.trams, route_id=route.config.route_id)
                tl.create_summary(route.trams, route_id=route.config.route_id)

        if plot_graphs:
            plots_dir = os.path.join(self.run_dir, "plots")
            for route in self.routes:
                route_plots = os.path.join(plots_dir, route.config.route_id)
                os.makedirs(route_plots, exist_ok=True)
                viz = TramVisualization(
                    {sid: self.shared_stops[sid] for sid in route.config.stop_ids
                     if sid in self.shared_stops},
                    route.config.simulation_hours,
                )
                viz.create_all_plots(trams=route.trams, output_dir=route_plots)

    # ── Метрики для оптимизатора ──────────────────────────────────────────────

    def get_objectives(self) -> Tuple[float, float]:
        """
        Возвращает (total_avg_waiting_time, total_tram_km) — две цели NSGA-II.
        Минимизируются обе.
        """
        total_wait = sum(
            s.avg_waiting_time * s.passengers_served
            for s in self.shared_stops.values()
            if s.passengers_served > 0
        )
        total_served = sum(
            s.passengers_served for s in self.shared_stops.values()
        )
        avg_wait = total_wait / total_served if total_served > 0 else 0.0
        total_km = sum(r.stats.total_tram_km for r in self.routes)
        return avg_wait, total_km

    def get_full_stats(self) -> dict:
        avg_wait, total_km = self.get_objectives()
        return {
            "routes": {
                r.config.route_id: {
                    "passengers_served": r.stats.total_passengers_served,
                    "tram_km": r.stats.total_tram_km,
                    "passenger_km": r.stats.total_passenger_km,
                    "avg_utilization_deviation": (
                        sum(r.stats.utilization_deviations) / len(r.stats.utilization_deviations)
                        if r.stats.utilization_deviations else 0.0
                    ),
                }
                for r in self.routes
            },
            "global": {
                "avg_waiting_time_min": avg_wait,
                "total_tram_km": total_km,
                "unique_stops": len(self.shared_stops),
            }
        }

    # ── Вывод статистики ──────────────────────────────────────────────────────

    def _print_stats(self):
        log.info(f"\n{'='*60}")
        log.info("РЕЗУЛЬТАТЫ МУЛЬТИМАРШРУТНОЙ СИМУЛЯЦИИ")
        log.info(f"{'='*60}")
        stats = self.get_full_stats()

        for route_id, rs in stats["routes"].items():
            log.info(f"\nМаршрут {route_id}:")
            log.info(f"  • Пассажиры:         {rs['passengers_served']}")
            log.info(f"  • Трамвай-км:        {rs['tram_km']:.1f}")
            log.info(f"  • Пассажиро-км:      {rs['passenger_km']:.1f}")
            log.info(f"  • Откл. загруженности: {rs['avg_utilization_deviation']:.2%}")

        g = stats["global"]
        log.info(f"\nГлобально:")
        log.info(f"  • Ср. время ожидания: {g['avg_waiting_time_min']:.2f} мин")
        log.info(f"  • Всего трамвай-км:   {g['total_tram_km']:.1f}")
        log.info(f"  • Уникальных ост.:    {g['unique_stops']}")
        log.info(f"{'='*60}\n")
