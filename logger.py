"""
Модуль логирования детальных событий трамваев.

Формат событий (ожидается в stop_log):
{
  "time": float,                 # минуты с начала симуляции
  "stop_id": int,
  "direction": "forward"|"backward",
  "waiting_before": int,
  "alighted": int,
  "boarded": int,
  "passengers_in_tram": int,
  "utilization_after": float     # обычно в процентах (0..100)
  # (опционально) "route_id": str|int
}
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Any


log = logging.getLogger(__name__)


class TramLogger:
    """Класс для сохранения CSV-логов по трамваям и сводной таблицы."""

    TRAM_LOG_COLUMNS = [
        "tram_id",
        "route_id",
        "time_min",
        "hour",
        "stop_id",
        "direction",
        "waiting_before",
        "alighted",
        "boarded",
        "passengers_in_tram",
        "utilization_percent",
    ]

    SUMMARY_COLUMNS = [
        "tram_id",
        "route_id",
        "total_trips",
        "passengers_served",
        "avg_utilization_percent",
        "max_utilization_percent",
        "stop_events_count",
    ]

    def __init__(
        self,
        output_dir: str = "tram_logs",
        file_prefix: str = "tram",
        write_header: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.file_prefix = file_prefix
        self.write_header = write_header
        self._ensure_output_dir()

    def _ensure_output_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _direction_label(direction: str) -> str:
        # Сохраняем машинно-удобный формат, без локализации.
        return direction if direction in ("forward", "backward") else str(direction)

    @staticmethod
    def _safe_int(x: Any, default: int = 0) -> int:
        try:
            return int(x)
        except Exception:
            return default

    @staticmethod
    def _safe_float(x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return default

    def save_tram_log(
        self,
        tram_id: int,
        stop_log: List[dict],
        route_id: Optional[str] = None,
    ) -> Optional[Path]:
        """Сохраняет лог остановок для одного трамвая в CSV.

        Returns:
            Path к файлу или None, если stop_log пуст.
        """
        if not stop_log:
            log.info(f"Tram #{tram_id}: no stop events to save")
            return None

        filename = f"{self.file_prefix}_{tram_id:03d}.csv"
        filepath = self.output_dir / filename

        with filepath.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.TRAM_LOG_COLUMNS)
            if self.write_header:
                writer.writeheader()

            for event in stop_log:
                t = self._safe_float(event.get("time", 0.0))
                writer.writerow({
                    "tram_id": tram_id,
                    "route_id": event.get("route_id", route_id),
                    "time_min": round(t, 4),
                    "hour": int(t // 60) % 24,
                    "stop_id": self._safe_int(event.get("stop_id", 0)),
                    "direction": self._direction_label(event.get("direction", "")),
                    "waiting_before": self._safe_int(event.get("waiting_before", 0)),
                    "alighted": self._safe_int(event.get("alighted", 0)),
                    "boarded": self._safe_int(event.get("boarded", 0)),
                    "passengers_in_tram": self._safe_int(event.get("passengers_in_tram", 0)),
                    "utilization_percent": round(self._safe_float(event.get("utilization_after", 0.0)), 4),
                })

        log.info(f"Tram #{tram_id}: {len(stop_log)} stop events → {filepath.name}")
        return filepath

    def save_all_trams(
        self,
        trams: Dict[int, Any],
        route_id: Optional[str] = None,
        include_empty: bool = False,
    ) -> List[Path]:
        """Сохраняет логи всех трамваев.

        Args:
            trams: dict tram_id -> tram object (ожидается tram.stats.stop_log).
            route_id: общий route_id для записи (если в event нет route_id).
            include_empty: если True, создаст пустые CSV с заголовком (обычно не нужно).

        Returns:
            Список путей к созданным файлам.
        """
        log.info("=" * 60)
        log.info("SAVING TRAM LOGS")
        log.info("=" * 60)

        paths: List[Path] = []
        for tram_id, tram in trams.items():
            stop_log = getattr(getattr(tram, "stats", None), "stop_log", None) or []

            if stop_log:
                p = self.save_tram_log(tram_id, stop_log, route_id=route_id)
                if p is not None:
                    paths.append(p)
            elif include_empty:
                # Пустой файл с заголовком (для фиксированного набора файлов)
                filename = f"{self.file_prefix}_{tram_id:03d}.csv"
                filepath = self.output_dir / filename
                with filepath.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=self.TRAM_LOG_COLUMNS)
                    if self.write_header:
                        writer.writeheader()
                paths.append(filepath)

        log.info("-" * 60)
        log.info(f"Saved logs: {len(paths)} / {len(trams)}")
        log.info(f"Folder: {self.output_dir.resolve()}")
        log.info("=" * 60)
        return paths

    def create_summary(
        self,
        trams: Dict[int, Any],
        output_file: str = "trams_summary.csv",
        route_id: Optional[str] = None,
    ) -> Path:
        """Создаёт сводную таблицу по всем трамваям.

        Ожидается:
          tram.stats.total_trips
          tram.stats.passengers_served
          tram.stats.utilization_history (0..1)
          tram.stats.stop_log
        """
        filepath = self.output_dir / output_file

        with filepath.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.SUMMARY_COLUMNS)
            writer.writeheader()

            for tram_id, tram in sorted(trams.items(), key=lambda x: x[0]):
                stats = getattr(tram, "stats", None)
                if stats is None:
                    continue

                util_hist: List[float] = list(getattr(stats, "utilization_history", []) or [])
                stop_log: List[dict] = list(getattr(stats, "stop_log", []) or [])

                if not util_hist and not stop_log:
                    continue

                avg_util = (sum(util_hist) / len(util_hist)) if util_hist else 0.0
                max_util = max(util_hist) if util_hist else 0.0

                # Пытаемся взять route_id из события, если он там есть.
                detected_route_id = None
                if stop_log:
                    detected_route_id = stop_log[0].get("route_id", None)

                writer.writerow({
                    "tram_id": tram_id,
                    "route_id": detected_route_id if detected_route_id is not None else route_id,
                    "total_trips": int(getattr(stats, "total_trips", 0) or 0),
                    "passengers_served": int(getattr(stats, "passengers_served", 0) or 0),
                    "avg_utilization_percent": round(avg_util * 100.0, 4),
                    "max_utilization_percent": round(max_util * 100.0, 4),
                    "stop_events_count": len(stop_log),
                })

        log.info(f"Summary saved → {filepath.name}")
        return filepath
