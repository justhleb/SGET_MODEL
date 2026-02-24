"""
Имитационная модель движения трамваев на SimPy
"""

import simpy
import random
import json
import math
import logging
import os
import argparse
from typing import Dict, List
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime

from visualization import TramVisualization
from logger import TramLogger


# ─── Константы ────────────────────────────────────────────────────────────────
CONFIG_DIR = "configs"
OUTPUT_DIR = "outputs"

BASE_ALIGHT_RATE     = 0.20  # Базовый процент высадки на остановке
PEAK_ALIGHT_MULT     = 2.0   # Множитель высадки на популярной остановке
END_BONUS_MAX        = 0.30  # Максимальный прогрессивный бонус к концу маршрута
MAX_ALIGHT_RATE      = 0.80  # Потолок доли высадки за одну остановку
MIN_SPEED_KMH        = 5.0   # Минимальная скорость трамвая (км/ч)
SPEED_VARIATION      = 0.05  # Случайное отклонение скорости ±5%
BOARDING_MIN_PER_PAX = 0.05  # Минут на посадку/высадку одного пассажира
DEFAULT_TURNAROUND   = 2.0   # Время разворота (мин), если не указано в конфиге
DEFAULT_TARGET_UTIL  = 0.75  # Целевая загруженность, если не указана в конфиге
DEFAULT_ROAD_LOAD    = 0.50  # Загруженность по умолчанию для часов без данных


# ─── Логирование ──────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ─── Классы данных ────────────────────────────────────────────────────────────

@dataclass
class TramStats:
    """Статистика трамвая"""
    tram_id: int
    passengers_served: int = 0
    total_trips: int = 0
    utilization_history: List[float] = field(default_factory=list)
    stop_log: List[dict] = field(default_factory=list)


class Stop:
    """Класс остановки"""

    def __init__(self, stop_id: int, env: simpy.Environment):
        self.stop_id = stop_id
        self.env = env
        self.waiting_passengers = 0
        self.last_tram_time = 0.0
        self.total_waiting_time = 0.0
        self.passengers_served = 0
        self.waiting_history: List[tuple] = []

    def record_waiting(self):
        """Записывает текущее количество ожидающих"""
        self.waiting_history.append((self.env.now, self.waiting_passengers))

    def get_new_passengers(self, intensity_per_hour: float, time_since_last: float) -> int:
        """Рассчитывает количество новых пассажиров (аппроксимация Пуассона)"""
        if time_since_last <= 0 or intensity_per_hour <= 0:
            return 0
        rate = intensity_per_hour * (time_since_last / 60.0)
        return max(0, int(random.gauss(rate, math.sqrt(rate))))

    def add_waiting_time(self, boarded: int, time_since_last: float):
        """Добавляет время ожидания для обслуженных пассажиров.

        Предполагает равномерное прибытие пассажиров:
        среднее время ожидания = половина интервала между трамваями.
        """
        self.total_waiting_time += boarded * (time_since_last / 2.0)
        self.passengers_served += boarded


class Tram:
    """Класс трамвая"""

    def __init__(self, tram_id: int, capacity: int):
        self.tram_id = tram_id
        self.capacity = capacity
        self.passengers = 0
        self.direction = "forward"
        self.stats = TramStats(tram_id)

    @property
    def free_seats(self) -> int:
        return self.capacity - self.passengers

    @property
    def utilization(self) -> float:
        return self.passengers / self.capacity if self.capacity > 0 else 0.0

    def board_passengers(self, waiting: int) -> int:
        """Посадка пассажиров"""
        can_board = min(waiting, self.free_seats)
        self.passengers += can_board
        return can_board

    def alight_passengers(self, stop_id: int, total_stops: int, peak_stop: int) -> int:
        """Высадка пассажиров"""
        if self.passengers == 0:
            return 0

        # Полная высадка на конечной остановке
        is_terminal = (
            (self.direction == "forward"  and stop_id == total_stops) or
            (self.direction == "backward" and stop_id == 1)
        )
        if is_terminal:
            alighted = self.passengers
            self.passengers = 0
            return alighted

        # Базовый процент + бонус на популярной остановке
        rate = BASE_ALIGHT_RATE
        if stop_id == peak_stop:
            rate = min(rate * PEAK_ALIGHT_MULT, MAX_ALIGHT_RATE)

        # Прогрессивный бонус к концу маршрута
        progress = (
            stop_id / total_stops
            if self.direction == "forward"
            else (total_stops - stop_id + 1) / total_stops
        )
        rate = min(rate + progress * END_BONUS_MAX, MAX_ALIGHT_RATE)

        alighted = int(self.passengers * rate)
        self.passengers -= alighted
        return alighted

    def log_stop_event(
        self,
        time: float,
        stop_id: int,
        direction: str,
        waiting_before: int,
        alighted: int,
        boarded: int,
        utilization_after: float,
    ):
        """Записывает событие прибытия на остановку"""
        self.stats.stop_log.append({
            "time": time,
            "stop_id": stop_id,
            "direction": direction,
            "waiting_before": waiting_before,
            "alighted": alighted,
            "boarded": boarded,
            "passengers_in_tram": self.passengers,
            "utilization_after": utilization_after,
        })


class TramSimulation:
    """Главный класс симуляции"""

    def __init__(self, config_file: str):
        if not os.path.exists(config_file):
            config_file = os.path.join(CONFIG_DIR, config_file)
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Конфигурация не найдена: {config_file}")

        self.config_file = config_file
        self.load_config(config_file)           # ← единственный вызов

        self.run_dir = self._create_run_directory()
        self.env = simpy.Environment()
        self.stops: Dict[int, Stop] = {}
        self.trams: Dict[int, Tram] = {}
        self.tram_counter = 0
        self.available_trams = simpy.Store(self.env)
        self.all_trams_created: List[Tram] = []

        self.stats = {
            "total_passengers_served": 0,
            "utilization_deviations": [],
            "total_tram_km": 0.0,
            "total_passenger_km": 0.0,
        }
        self.setup_stops()

    # ── Конфигурация ──────────────────────────────────────────────────────────

    def load_config(self, config_file: str):
        """Загружает конфигурацию из JSON"""
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.stop_number          = config["stop_number"]
        self.flow_speed           = config["flow_speed"]
        self.peak_stop            = config["peak_stop"]
        self.tram_capacity        = config["tram_capacity"]
        self.simulation_hours     = config["simulation_hours"]

        # Параметры с дефолтами (обратная совместимость со старыми конфигами)
        self.tram_count           = config.get("tram_count", 8)
        self.acceleration_time    = config.get("acceleration_time", 0.5)
        self.stop_time            = config.get("stop_time", 1.0)
        self.turnaround_time      = config.get("turnaround_time", DEFAULT_TURNAROUND)
        self.target_utilization   = config.get("target_utilization", DEFAULT_TARGET_UTIL)
        self.operation_start_hour = config.get("operation_start_hour", 6)
        self.operation_end_hour   = config.get("operation_end_hour", 24)
        self.random_seed          = config.get("random_seed", None)

        self.distances = {item[0]: item[1] for item in config["distance"]}

        self.intensity_map: Dict[int, Dict[int, float]] = defaultdict(dict)
        for stop_id, hour, intensity in config["intensity"]:
            self.intensity_map[stop_id][hour] = intensity

        self.bus_intervals = sorted(config["bus_interval"], key=lambda x: x[0])
        self.road_loads    = {hour: load for hour, load in config["road_loads"]}

    # ── Инициализация ─────────────────────────────────────────────────────────

    def _create_run_directory(self) -> str:
        """Создаёт уникальную папку для результатов запуска"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = os.path.join(OUTPUT_DIR, f"run_{timestamp}")
        os.makedirs(os.path.join(run_dir, "logs"),  exist_ok=True)
        os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
        log.info(f"Результаты будут сохранены в: {run_dir}/")
        return run_dir

    def setup_stops(self):
        """Инициализирует остановки"""
        for i in range(1, self.stop_number + 1):
            self.stops[i] = Stop(i, self.env)

    def create_initial_trams(self):
        """Создаёт начальный парк трамваев (размер берётся из конфига)"""
        for _ in range(self.tram_count):
            self.tram_counter += 1
            tram = Tram(self.tram_counter, self.tram_capacity)
            self.trams[self.tram_counter] = tram
            self.all_trams_created.append(tram)
            self.available_trams.put(tram)
        log.info(f"Создан парк из {self.tram_count} трамваев")

    # ── Вспомогательные методы ────────────────────────────────────────────────

    def get_intensity(self, stop_id: int, hour: int) -> float:
        """Получает интенсивность для остановки и часа"""
        return self.intensity_map.get(stop_id, {}).get(hour, 0)

    def get_current_interval(self, current_time: float) -> int:
        """Получает текущий интервал между выходами трамваев (мин)"""
        hour = int(current_time // 60) % 24
        for start_hour, interval in reversed(self.bus_intervals):
            if hour >= start_hour:
                return interval
        return self.bus_intervals[0][1]

    def get_road_load(self, hour: int) -> float:
        """Возвращает загруженность дороги с линейной интерполяцией между часами.

        Если час точно есть в конфиге — возвращает его значение напрямую.
        Иначе интерполирует между ближайшими соседями.
        Для часов вне диапазона — возвращает крайнее значение.
        """
        if hour in self.road_loads:
            return self.road_loads[hour]

        hours = sorted(self.road_loads)
        if not hours:
            return DEFAULT_ROAD_LOAD

        prev_hours = [h for h in hours if h <= hour]
        next_hours = [h for h in hours if h > hour]

        if not prev_hours:
            return self.road_loads[hours[0]]
        if not next_hours:
            return self.road_loads[hours[-1]]

        h0, h1 = prev_hours[-1], next_hours[0]
        t = (hour - h0) / (h1 - h0)
        return self.road_loads[h0] * (1 - t) + self.road_loads[h1] * t

    def calculate_travel_time(self, distance: float, hour: int) -> float:
        """Рассчитывает время поездки между остановками (в минутах)"""
        if distance <= 0:
            return 0.0

        load_factor = self.get_road_load(hour)
        speed = self.flow_speed * (1.0 - load_factor)
        speed *= random.uniform(1.0 - SPEED_VARIATION, 1.0 + SPEED_VARIATION)
        speed = max(speed, MIN_SPEED_KMH)

        # время (мин) = дистанция (м) / 1000 * 60 / скорость (км/ч)
        travel_time = (distance / 1000.0) * (60.0 / speed)
        return travel_time + self.acceleration_time

    def _is_operating_hour(self, hour: int) -> bool:
        """Проверяет, входит ли час в рабочее время"""
        if self.operation_end_hour > self.operation_start_hour:
            return self.operation_start_hour <= hour < self.operation_end_hour
        # Ночной режим (напр. 22–6)
        return hour >= self.operation_start_hour or hour < self.operation_end_hour

    def _minutes_until_operation_start(self, current_hour: int) -> float:
        """Рассчитывает минуты до начала работы"""
        if current_hour < self.operation_start_hour:
            hours_to_wait = self.operation_start_hour - current_hour
        else:
            hours_to_wait = (24 - current_hour) + self.operation_start_hour
        current_minute = self.env.now % 60
        return hours_to_wait * 60 - current_minute

    # ── Процессы SimPy ────────────────────────────────────────────────────────

    def tram_generator(self):
        """Генератор выхода трамваев с соблюдением расписания.

        Исправленная логика: плановое время следующего выхода фиксируется
        ДО ожидания трамвая из пула. Если трамвай ждали дольше интервала —
        следующий выходит немедленно, расписание не «сдвигается» накопительно.
        """
        while True:
            current_hour = int(self.env.now // 60) % 24

            if not self._is_operating_hour(current_hour):
                wait = self._minutes_until_operation_start(current_hour)
                if wait > 0:
                    yield self.env.timeout(wait)
                continue

            interval = self.get_current_interval(self.env.now)
            next_departure = self.env.now + interval   # плановый выход фиксируется здесь

            tram = yield self.available_trams.get()    # ждём свободный трамвай

            log.info(
                f"[{self.env.now:.1f}] Трамвай #{tram.tram_id} выехал на маршрут "
                f"(рейс #{tram.stats.total_trips + 1})"
            )
            self.env.process(self.tram_process(tram))

            # Ждём только оставшееся до планового времени.
            # Если парк был занят дольше интервала — следующий выходит сразу.
            remaining = next_departure - self.env.now
            if remaining > 0:
                yield self.env.timeout(remaining)

    def arrive_at_stop(self, tram: Tram, stop_id: int):
        """Процесс прибытия на остановку"""
        stop = self.stops[stop_id]
        hour = int(self.env.now // 60) % 24
        time_since_last = self.env.now - stop.last_tram_time

        waiting_before = stop.waiting_passengers

        # Высадка
        alighted = tram.alight_passengers(stop_id, self.stop_number, self.peak_stop)

        # Новые пассажиры на остановке
        new_passengers = stop.get_new_passengers(
            self.get_intensity(stop_id, hour), time_since_last
        )
        stop.waiting_passengers += new_passengers
        stop.record_waiting()   # снимок ДО посадки

        # Посадка
        boarded = tram.board_passengers(stop.waiting_passengers)
        stop.waiting_passengers -= boarded
        stop.record_waiting()   # снимок ПОСЛЕ посадки

        # Обновление статистики
        if boarded > 0:
            stop.add_waiting_time(boarded, time_since_last)
        stop.last_tram_time = self.env.now
        tram.stats.passengers_served += boarded
        self.stats["total_passengers_served"] += boarded

        tram.stats.utilization_history.append(tram.utilization)
        self.stats["utilization_deviations"].append(
            abs(tram.utilization - self.target_utilization)
        )

        tram.log_stop_event(
            time=self.env.now,
            stop_id=stop_id,
            direction=tram.direction,
            waiting_before=waiting_before + new_passengers,
            alighted=alighted,
            boarded=boarded,
            utilization_after=tram.utilization * 100,
        )

        boarding_time = (boarded + alighted) * BOARDING_MIN_PER_PAX
        yield self.env.timeout(self.stop_time + boarding_time)

    def tram_process(self, tram: Tram):
        """Процесс движения трамвая по маршруту (один полный рейс туда-обратно)"""
        try:
            tram.stats.total_trips += 1

            for direction in ("forward", "backward"):
                tram.direction = direction
                stops_sequence = (
                    list(range(1, self.stop_number + 1))
                    if direction == "forward"
                    else list(range(self.stop_number, 0, -1))
                )

                for i, stop_id in enumerate(stops_sequence):
                    if i > 0:
                        prev_stop = stops_sequence[i - 1]
                        distance = (
                            self.distances.get(stop_id, 0)
                            if direction == "forward"
                            else self.distances.get(prev_stop, 0)
                        )
                        hour = int(self.env.now // 60) % 24
                        travel_time = self.calculate_travel_time(distance, hour)

                        distance_km = distance / 1000.0
                        self.stats["total_tram_km"]      += distance_km
                        self.stats["total_passenger_km"] += distance_km * tram.passengers

                        yield self.env.timeout(travel_time)

                    yield self.env.process(self.arrive_at_stop(tram, stop_id))

                if direction == "forward":
                    yield self.env.timeout(self.turnaround_time)

            log.info(
                f"[{self.env.now:.1f}] Трамвай #{tram.tram_id} вернулся на базу. "
                f"Рейс #{tram.stats.total_trips}, "
                f"обслужено {tram.stats.passengers_served} пас. (всего)"
            )
            yield self.available_trams.put(tram)

        except simpy.Interrupt:
            log.warning(f"[{self.env.now:.1f}] Трамвай #{tram.tram_id} прерван")

    # ── Запуск и результаты ───────────────────────────────────────────────────

    def run_simulation(self, plot_graphs: bool = True, save_logs: bool = True):
        """Запуск симуляции"""
        if self.random_seed is not None:
            random.seed(self.random_seed)
            log.info(f"Случайный seed: {self.random_seed}")

        log.info(f"\n{'='*60}")
        log.info("Запуск симуляции движения трамваев")
        log.info(f"Конфигурация:        {os.path.basename(self.config_file)}")
        log.info(f"Длительность:        {self.simulation_hours} часов")
        log.info(f"Количество остановок:{self.stop_number}")
        log.info(f"Вместимость трамвая: {self.tram_capacity} чел.")
        log.info(f"Парк трамваев:       {self.tram_count} ед.")
        log.info(f"{'='*60}\n")

        self.create_initial_trams()
        self.env.process(self.tram_generator())
        self.env.run(until=self.simulation_hours * 60)
        self.print_final_stats()

        if save_logs:
            logs_dir = os.path.join(self.run_dir, "logs")
            tram_logger = TramLogger(output_dir=logs_dir)
            tram_logger.save_all_trams(self.trams)
            tram_logger.create_summary(self.trams)

        if plot_graphs:     # ← теперь флаг работает корректно
            plots_dir = os.path.join(self.run_dir, "plots")
            viz = TramVisualization(self.stops, self.simulation_hours)
            viz.create_all_plots(trams=self.trams, output_dir=plots_dir)

        log.info(f"\n{'='*60}")
        log.info("СИМУЛЯЦИЯ ЗАВЕРШЕНА")
        log.info(f"Результаты сохранены: {self.run_dir}/")
        log.info(f"{'='*60}\n")

    def print_final_stats(self):
        """Вывод финальной статистики"""
        log.info(f"\n{'='*60}")
        log.info("РЕЗУЛЬТАТЫ СИМУЛЯЦИИ")
        log.info(f"{'='*60}")
        log.info(f"Время симуляции: {self.simulation_hours} часов ({self.env.now:.1f} мин)")

        log.info("\nОбщая статистика:")
        log.info(f"  • Всего обслужено пассажиров: {self.stats['total_passengers_served']}")
        log.info(f"  • Трамваев в парке:           {len(self.all_trams_created)}")
        log.info(f"  • Трамвай-километры:          {self.stats['total_tram_km']:.1f} км")
        log.info(f"  • Пассажиро-километры:        {self.stats['total_passenger_km']:.1f} пас⋅км")

        if self.stats["utilization_deviations"]:
            avg_dev = (
                sum(self.stats["utilization_deviations"])
                / len(self.stats["utilization_deviations"])
            )
            log.info(
                f"  • Среднее отклонение загруженности "
                f"от {self.target_utilization:.0%}: {avg_dev:.2%}"
            )

        log.info("\nСтатистика по трамваям:")
        active = sorted(
            [t for t in self.all_trams_created if t.stats.total_trips > 0],
            key=lambda t: t.tram_id,
        )
        for tram in active:
            if tram.stats.utilization_history:
                avg_util = (
                    sum(tram.stats.utilization_history)
                    / len(tram.stats.utilization_history)
                )
                log.info(
                    f"  Трамвай #{tram.tram_id}: {tram.stats.passengers_served} пас., "
                    f"{tram.stats.total_trips} рейсов, средняя загрузка {avg_util:.1%}"
                )

        log.info("\nСтатистика по остановкам:")
        for stop_id in sorted(self.stops):
            stop = self.stops[stop_id]
            if stop.passengers_served > 0:
                avg_wait = stop.total_waiting_time / stop.passengers_served
                log.info(
                    f"  Остановка {stop_id:2d}: обслужено {stop.passengers_served:4d} пас., "
                    f"среднее ожидание {avg_wait:5.1f} мин, "
                    f"осталось {stop.waiting_passengers} в очереди"
                )

        log.info(f"{'='*60}\n")


# ─── Точка входа ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Симуляция движения трамваев")
    parser.add_argument(
        "--config", type=str, default="tram_config.json",
        help="Имя файла конфигурации (из папки configs/)",
    )
    parser.add_argument("--no-plots", action="store_true", help="Не создавать графики")
    parser.add_argument("--no-logs",  action="store_true", help="Не сохранять детальные логи")
    args = parser.parse_args()

    sim = TramSimulation(args.config)
    sim.run_simulation(
        plot_graphs=not args.no_plots,
        save_logs=not args.no_logs,
    )


if __name__ == "__main__":
    main()
