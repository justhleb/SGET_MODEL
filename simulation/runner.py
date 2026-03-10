"""
Точка входа CLI для мультимаршрутной симуляции.

Примеры:
    python -m simulation.runner --routes 20 48 55
    python -m simulation.runner --routes 20 --no-plots
    python -m simulation.runner --routes 20 48 --configs-dir my_configs
"""
import argparse
import logging
import sys
from pathlib import Path

from simulation.multi_route import MultiRouteSimulation

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

DEFAULT_CONFIGS_DIR = Path("configs")


def resolve_config_files(route_ids: list[str], configs_dir: Path) -> list[str]:
    """
    По номеру маршрута ищет fwd и bwd конфиги в configs_dir.
    Паттерн: route_{id}_fwd_config.json и route_{id}_bwd_config.json
    """
    resolved = []
    missing  = []

    for route_id in route_ids:
        for direction in ("fwd", "bwd"):
            path = configs_dir / f"route_{route_id}_{direction}_config.json"
            if path.exists():
                resolved.append(str(path))
            else:
                missing.append(str(path))

    if missing:
        log.error("Не найдены конфиги:")
        for m in missing:
            log.error(f"  ✗ {m}")
        sys.exit(1)

    log.info(f"Найдено конфигов: {len(resolved)}")
    for p in resolved:
        log.info(f"  ✓ {p}")

    return resolved


def main():
    parser = argparse.ArgumentParser(
        description="Мультимаршрутная симуляция трамваев",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Примеры использования:\n"
            "  python -m simulation.runner --routes 20 48 55\n"
            "  python -m simulation.runner --routes 20 --no-plots\n"
            "  python -m simulation.runner --routes 20 48 --configs-dir my_configs\n"
        )
    )
    parser.add_argument(
        "--routes", nargs="+", required=True, metavar="ROUTE_ID",
        help="Номера маршрутов для симуляции (например: 20 48 55)"
    )
    parser.add_argument(
        "--configs-dir", type=Path, default=DEFAULT_CONFIGS_DIR, metavar="DIR",
        help=f"Папка с конфиг-файлами (по умолчанию: {DEFAULT_CONFIGS_DIR})"
    )
    parser.add_argument("--no-plots", action="store_true", help="Не создавать графики")
    parser.add_argument("--no-logs",  action="store_true", help="Не сохранять CSV-логи")
    args = parser.parse_args()

    if not args.configs_dir.exists():
        log.error(f"Папка конфигов не найдена: {args.configs_dir}")
        sys.exit(1)

    config_files = resolve_config_files(args.routes, args.configs_dir)

    sim = MultiRouteSimulation(config_files)
    sim.run(
        plot_graphs=not args.no_plots,
        save_logs=not args.no_logs,
    )


if __name__ == "__main__":
    main()