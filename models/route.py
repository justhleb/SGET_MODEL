"""
RouteConfig — чистый датакласс конфига маршрута (загрузка из JSON).
Route       — SimPy-процессы одного маршрута в общем env.
"""
from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import simpy

from models.stop import Stop, StopEvent
from models.tram import Tram, BOARDING_MIN_PER_PAX

log = logging.getLogger(__name__)

# ── Дефолты ───────────────────────────────────────────────────────────────────
DEFAULT_TURNAROUND  = 2.0
DEFAULT_TARGET_UTIL = 0.75
DEFAULT_ROAD_LOAD   = 0.50
MIN_SPEED_KMH       = 5.0
SPEED_VARIATION     = 0.05


@dataclass
class RouteStats:
    route_id: str
    total_passengers_served: int = 0
    total_tram_km: float = 0.0
    total_passenger_km: float = 0.0
    utilization_deviations: List[float] = field(default_factory=list)


@dataclass
class RouteConfig:
    route_id: str
    stop_ids: List[int]          # глобальные ID остановок в порядке маршрута
    tram_count: int
    tram_capacity: int
    flow_speed: float            # базовая скорость, км/ч
    peak_stop_index: int         # 1-based позиция популярной остановки в stop_ids
    simulation_hours: int
    distances: Dict[int, float]  # {stop_id: расстояние от предыдущей, м}
    intensity_map: Dict[int, Dict[int, float]]   # {stop_id: {hour: pax/h}}
    bus_intervals: List[Tuple[int, int]]         # [(start_hour, interval_min), ...]
    road_loads: Dict[int, float]                 # {hour: 0..1}
    turnaround_time: float = DEFAULT_TURNAROUND
    acceleration_time: float = 0.5
    stop_time: float = 1.0
    target_utilization: float = DEFAULT_TARGET_UTIL
    operation_start_hour: int = 6
    operation_end_hour: int = 24
    random_seed: Optional[int] = None

    @property
    def stop_number(self) -> int:
        return len(self.stop_ids)

    @classmethod
    def from_json(cls, config_file: str) -> "RouteConfig":
        with open(config_file, "r", encoding="utf-8") as f:
            c = json.load(f)

        distances = {item[0]: item[1] for item in c["distance"]}

        intensity_map: Dict[int, Dict[int, float]] = defaultdict(dict)
        for stop_id, hour, intensity in c["intensity"]:
            intensity_map[stop_id][hour] = intensity

        bus_intervals = sorted(c["bus_interval"], key=lambda x: x[0])
        road_loads = {hour: load for hour, load in c["road_loads"]}

        # Обратная совместимость: если stop_ids нет — генерим из stop_number
        stop_ids = c.get("stop_ids", list(range(1, c["stop_number"] + 1)))

        # peak_stop в старом формате — глобальный ID; переводим в индекс
        raw_peak = c.get("peak_stop", stop_ids[len(stop_ids) // 2])
        if raw_peak in stop_ids:
            peak_stop_index = stop_ids.index(raw_peak) + 1  # 1-based
        else:
            peak_stop_index = len(stop_ids) // 2

        return cls(
            route_id=str(c.get("route_id", config_file)),
            stop_ids=stop_ids,
            tram_count=c.get("tram_count", 8),
            tram_capacity=c["tram_capacity"],
            flow_speed=c["flow_speed"],
            peak_stop_index=peak_stop_index,
            simulation_hours=c["simulation_hours"],
            distances=distances,
            intensity_map=dict(intensity_map),
            bus_intervals=bus_intervals,
            road_loads=road_loads,
            turnaround_time=c.get("turnaround_time", DEFAULT_TURNAROUND),
            acceleration_time=c.get("acceleration_time", 0.5),
            stop_time=c.get("stop_time", 1.0),
            target_utilization=c.get("target_utilization", DEFAULT_TARGET_UTIL),
            operation_start_hour=c.get("operation_start_hour", 6),
            operation_end_hour=c.get("operation_end_hour", 24),
            random_seed=c.get("random_seed", None),
        )


class Route:
    """
    Один маршрут в общем simpy.Environment.
    Не создаёт env и остановки — получает их снаружи.
    """

    def __init__(
        self,
        config: RouteConfig,
        env: simpy.Environment,
        shared_stops: Dict[int, Stop],
        tram_id_offset: int = 0,    # чтобы ID трамваев были уникальны глобально
    ):
        self.config = config
        self.env = env
        self.shared_stops = shared_stops
        self.tram_id_offset = tram_id_offset

        self.trams: Dict[int, Tram] = {}
        self.all_trams: List[Tram] = []
        self.available_trams: simpy.Store = simpy.Store(env)
        self.stats = RouteStats(route_id=config.route_id)
        self._tram_counter = tram_id_offset

    # ── Инициализация ─────────────────────────────────────────────────────────

    def _spawn_trams(self):
        for _ in range(self.config.tram_count):
            self._tram_counter += 1
            tram = Tram(self._tram_counter, self.config.route_id, self.config.tram_capacity)
            self.trams[self._tram_counter] = tram
            self.all_trams.append(tram)
            self.available_trams.put(tram)
        log.info(f"[Route {self.config.route_id}] Создан парк: {self.config.tram_count} трамваев")

    def start(self):
        """Регистрирует процессы маршрута в env. Вызывается из MultiRouteSimulation."""
        self._spawn_trams()
        self.env.process(self._tram_generator())

    # ── Вспомогательные ───────────────────────────────────────────────────────

    def _get_intensity(self, stop_id: int, hour: int) -> float:
        return self.config.intensity_map.get(stop_id, {}).get(hour, 0.0)

    def _get_current_interval(self, current_time: float) -> int:
        hour = int(current_time // 60) % 24
        for start_hour, interval in reversed(self.config.bus_intervals):
            if hour >= start_hour:
                return interval
        return self.config.bus_intervals[0][1]

    def _get_road_load(self, hour: int) -> float:
        rl = self.config.road_loads
        if hour in rl:
            return rl[hour]
        hours = sorted(rl)
        if not hours:
            return DEFAULT_ROAD_LOAD
        prev = [h for h in hours if h <= hour]
        nxt  = [h for h in hours if h > hour]
        if not prev:
            return rl[hours[0]]
        if not nxt:
            return rl[hours[-1]]
        h0, h1 = prev[-1], nxt[0]
        t = (hour - h0) / (h1 - h0)
        return rl[h0] * (1 - t) + rl[h1] * t

    def _calculate_travel_time(self, distance: float, hour: int) -> float:
        if distance <= 0:
            return 0.0
        load = self._get_road_load(hour)
        speed = self.config.flow_speed * (1.0 - load)
        speed *= random.uniform(1.0 - SPEED_VARIATION, 1.0 + SPEED_VARIATION)
        speed = max(speed, MIN_SPEED_KMH)
        return (distance / 1000.0) * (60.0 / speed) + self.config.acceleration_time

    def _is_operating(self, hour: int) -> bool:
        s, e = self.config.operation_start_hour, self.config.operation_end_hour
        if e > s:
            return s <= hour < e
        return hour >= s or hour < e

    def _minutes_until_start(self, current_hour: int) -> float:
        s = self.config.operation_start_hour
        hours_to_wait = (s - current_hour) if current_hour < s else (24 - current_hour + s)
        return hours_to_wait * 60 - (self.env.now % 60)

    # ── SimPy-процессы ────────────────────────────────────────────────────────

    def _tram_generator(self):
        while True:
            hour = int(self.env.now // 60) % 24
            if not self._is_operating(hour):
                wait = self._minutes_until_start(hour)
                if wait > 0:
                    yield self.env.timeout(wait)
                continue

            interval = self._get_current_interval(self.env.now)
            next_departure = self.env.now + interval

            tram = yield self.available_trams.get()
            log.info(
                f"[{self.env.now:.1f}] Маршрут {self.config.route_id}: "
                f"трамвай #{tram.tram_id} выехал (рейс #{tram.stats.total_trips + 1})"
            )
            self.env.process(self._tram_process(tram))

            remaining = next_departure - self.env.now
            if remaining > 0:
                yield self.env.timeout(remaining)

    def _arrive_at_stop(self, tram: Tram, stop_index: int):
        """
        stop_index — 1-based позиция в маршруте (не глобальный stop_id).
        """
        stop_id = self.config.stop_ids[stop_index - 1]
        stop = self.shared_stops[stop_id]
        hour = int(self.env.now // 60) % 24
        time_since_last = self.env.now - stop.last_tram_time

        waiting_before = stop.waiting_passengers

        # Высадка
        alighted = tram.alight_passengers(
            stop_index, self.config.stop_number, self.config.peak_stop_index
        )

        # Новые пассажиры
        new_pax = stop.get_new_passengers(
            self._get_intensity(stop_id, hour), time_since_last
        )
        stop.waiting_passengers += new_pax
        stop.record_waiting()

        # Посадка
        boarded = tram.board_passengers(stop.waiting_passengers)
        stop.waiting_passengers -= boarded
        stop.record_waiting()

        # Статистика
        if boarded > 0:
            stop.add_waiting_time(boarded, time_since_last)
        stop.last_tram_time = self.env.now

        tram.stats.passengers_served += boarded
        self.stats.total_passengers_served += boarded
        tram.stats.utilization_history.append(tram.utilization)
        self.stats.utilization_deviations.append(
            abs(tram.utilization - self.config.target_utilization)
        )

        tram.log_stop_event(
            time=self.env.now,
            stop_id=stop_id,
            direction=tram.direction,
            waiting_before=waiting_before + new_pax,
            alighted=alighted,
            boarded=boarded,
            utilization_after=tram.utilization * 100,
        )
        stop.log_event(StopEvent(
            time=self.env.now,
            route_id=self.config.route_id,
            tram_id=tram.tram_id,
            direction=tram.direction,
            waiting_before=waiting_before + new_pax,
            alighted=alighted,
            boarded=boarded,
            passengers_in_tram=tram.passengers,
            utilization_after=tram.utilization,
        ))

        boarding_time = (boarded + alighted) * BOARDING_MIN_PER_PAX
        yield self.env.timeout(self.config.stop_time + boarding_time)

    def _tram_process(self, tram: Tram):
        try:
            tram.stats.total_trips += 1
            cfg = self.config

            for direction in ("forward", "backward"):
                tram.direction = direction
                indices = (
                    list(range(1, cfg.stop_number + 1))
                    if direction == "forward"
                    else list(range(cfg.stop_number, 0, -1))
                )

                for i, stop_index in enumerate(indices):
                    if i > 0:
                        prev_index = indices[i - 1]
                        # distance привязана к stop_id, берём по направлению
                        ref_stop_id = (
                            cfg.stop_ids[stop_index - 1]
                            if direction == "forward"
                            else cfg.stop_ids[prev_index - 1]
                        )
                        distance = cfg.distances.get(ref_stop_id, 0.0)
                        hour = int(self.env.now // 60) % 24
                        travel_time = self._calculate_travel_time(distance, hour)

                        km = distance / 1000.0
                        self.stats.total_tram_km      += km
                        self.stats.total_passenger_km += km * tram.passengers

                        yield self.env.timeout(travel_time)

                    yield self.env.process(self._arrive_at_stop(tram, stop_index))

                if direction == "forward":
                    yield self.env.timeout(cfg.turnaround_time)

            log.info(
                f"[{self.env.now:.1f}] Маршрут {cfg.route_id}: "
                f"трамвай #{tram.tram_id} вернулся, рейс #{tram.stats.total_trips}, "
                f"обслужено {tram.stats.passengers_served} пас. (всего)"
            )
            yield self.available_trams.put(tram)

        except simpy.Interrupt:
            log.warning(f"Трамвай #{tram.tram_id} (маршрут {self.config.route_id}) прерван")
