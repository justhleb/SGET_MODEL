"""
Имитационная модель движения трамваев на SimPy
"""

import simpy
import random
import json
import math
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from visualization import TramVisualization
from logger import TramLogger
import os
from datetime import datetime

CONFIG_DIR = "configs"
OUTPUT_DIR = "outputs"

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
        self.waiting_history = []

    def record_waiting(self):
        """Записывает текущее количество ожидающих"""
        self.waiting_history.append((self.env.now, self.waiting_passengers))   
        
    def get_new_passengers(self, intensity_per_hour: float, time_since_last: float) -> int:
        """Рассчитывает количество новых пассажиров"""
        if time_since_last <= 0:
            return 0
        # Пуассоновское распределение
        rate = intensity_per_hour * (time_since_last / 60.0)
        if rate <= 0:
            return 0
        # Используем нормальное приближение для пуассона
        return max(0, int(random.gauss(rate, math.sqrt(rate))))
    
    def add_waiting_time(self, boarded: int, time_since_last: float):
        """Добавляет время ожидания для обслуженных пассажиров (точная версия)"""
        # Если пассажиры приходят равномерно, среднее время ожидания = половина интервала
        avg_individual_wait = time_since_last / 2.0
        self.total_waiting_time += boarded * avg_individual_wait
        self.passengers_served += boarded



class Tram:
    """Класс трамвая"""
    def __init__(self, tram_id: int, capacity: int):
        self.tram_id = tram_id
        self.capacity = capacity
        self.passengers = 0
        self.direction = "forward"  # forward или backward в зависимости от того куда едет трамвай
        self.stats = TramStats(tram_id)
        stop_log: List[dict] = field(default_factory=list)
        
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
            
        base_rate = 0.2
        
        # Больше выходят на популярной остановке
        if stop_id == peak_stop:
            base_rate *= 2.0
        
        # Больше выходят ближе к концу маршрута
        if self.direction == "forward":
            progress = stop_id / total_stops
        else:
            progress = (total_stops - stop_id + 1) / total_stops
        
        end_bonus = progress * 0.3
        alighting_rate = min(base_rate + end_bonus, 0.8)
        alighting = int(self.passengers * alighting_rate)
        
        # На конечной все выходят
        is_terminal = (self.direction == "forward" and stop_id == total_stops) or \
                      (self.direction == "backward" and stop_id == 1)
        
        if is_terminal:
            alighting = self.passengers
            
        self.passengers -= alighting
        return alighting
    
    def log_stop_event(self, time: float, stop_id: int, direction: str,
                       waiting_before: int, alighted: int, boarded: int, 
                       utilization_after: float):
        """Записывает событие прибытия на остановку"""
        self.stats.stop_log.append({
            'time': time,
            'stop_id': stop_id,
            'direction': direction,
            'waiting_before': waiting_before,
            'alighted': alighted,
            'boarded': boarded,
            'utilization_after': utilization_after
        })


class TramSimulation:
    """Главный класс симуляции"""
    
    def __init__(self, config_file: str):
        if not os.path.exists(config_file):
            config_file = os.path.join(CONFIG_DIR, config_file)
        
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Конфигурация не найдена: {config_file}")
        self.config_file = config_file
        self.load_config(config_file)
        self.run_dir = self._create_run_directory()

        self.load_config(config_file)
        self.env = simpy.Environment()
        self.stops: Dict[int, Stop] = {}
        self.trams: Dict[int, Tram] = {}
        self.tram_counter = 0
        
        # пул доступных трамваев
        self.available_trams = simpy.Store(self.env)
        self.all_trams_created = []
        
        self.stats = {
            'total_passengers_served': 0,
            'utilization_deviations': [],
            'waiting_time_deviations': [],
        }
        self.setup_stops()
    def _create_run_directory(self) -> str:
        """Создаёт уникальную папку для результатов запуска"""
        # Формат: outputs/run_2025-10-22_11-30-45/
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = os.path.join(OUTPUT_DIR, f"run_{timestamp}")
        
        # Создаём структуру папок
        os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
        
        print(f"Результаты будут сохранены в: {run_dir}/")
        return run_dir
        
    def load_config(self, config_file: str):
        """Загружает конфигурацию из JSON"""
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.stop_number = config['stop_number']
        self.distances = {item[0]: item[1] for item in config['distance']}
        self.flow_speed = config['flow_speed']
        self.peak_stop = config['peak_stop']
        self.tram_capacity = config['tram_capacity']
        self.simulation_hours = config['simulation_hours']
        self.acceleration_time = config.get('acceleration_time', 0.5)
        self.stop_time = config.get('stop_time', 1.0)
        
        self.intensity_map = defaultdict(dict)
        for stop_id, hour, intensity in config['intensity']:
            self.intensity_map[stop_id][hour] = intensity
            
        self.bus_intervals = sorted(config['bus_interval'], key=lambda x: x[0])
        self.road_loads = {hour: load for hour, load in config['road_loads']}
        
    def setup_stops(self):
        """Инициализирует остановки"""
        for i in range(1, self.stop_number + 1):
            self.stops[i] = Stop(i, self.env)
    
    def get_intensity(self, stop_id: int, hour: int) -> float:
        """Получает интенсивность для остановки и часа"""
        return self.intensity_map.get(stop_id, {}).get(hour, 0)
    
    def get_current_interval(self, current_time: float) -> int:
        """Получает текущий интервал между трамваями"""
        hour = int(current_time // 60) % 24
        for i in range(len(self.bus_intervals) - 1, -1, -1):
            if hour >= self.bus_intervals[i][0]:
                return self.bus_intervals[i][1]
        return self.bus_intervals[0][1]
    
    def calculate_travel_time(self, distance: float, hour: int) -> float:
        """Рассчитывает время поездки между остановками"""
        if distance <= 0:
            return 0
            
        # Базовая скорость с учётом загруженности дороги
        load_factor = self.road_loads.get(hour, 0.5)
        actual_speed = self.flow_speed * (1 - load_factor)
        
        # Добавляем случайное отклонение ±5%
        speed_variation = random.uniform(0.95, 1.05)
        actual_speed *= speed_variation
        
        # Минимальная скорость 5 км/ч
        actual_speed = max(actual_speed, 5.0)
        
        # Время = расстояние (м) / скорость (км/ч) * 60 (мин/ч) / 1000 (м/км)
        travel_time = (distance / 1000.0) * (60.0 / actual_speed)
        
        # Добавляем время разгона/торможения
        travel_time += self.acceleration_time
        
        return travel_time
    
    def arrive_at_stop(self, tram: Tram, stop_id: int):
        """Процесс прибытия на остановку"""
        stop = self.stops[stop_id]
        hour = int(self.env.now // 60) % 24
        
        # Запоминаем количество ожидающих ДО прибытия
        waiting_before = stop.waiting_passengers
        
        # Высадка пассажиров
        alighted = tram.alight_passengers(stop_id, self.stop_number, self.peak_stop)
        
        # Рассчитываем новых пассажиров на остановке
        time_since_last = self.env.now - stop.last_tram_time
        intensity = self.get_intensity(stop_id, hour)
        new_passengers = stop.get_new_passengers(intensity, time_since_last)
        stop.waiting_passengers += new_passengers
        
        # Записываем ДО посадки
        stop.record_waiting()
        
        # Посадка пассажиров
        boarded = tram.board_passengers(stop.waiting_passengers)
        stop.waiting_passengers -= boarded
        
        # Записываем ПОСЛЕ посадки
        stop.record_waiting()
        
        # Обновление статистики
        if boarded > 0:
            stop.add_waiting_time(boarded, time_since_last)
        stop.last_tram_time = self.env.now
        tram.stats.passengers_served += boarded
        self.stats['total_passengers_served'] += boarded
        
        # Записываем загруженность
        tram.stats.utilization_history.append(tram.utilization)
        utilization_deviation = abs(tram.utilization - 0.75)
        self.stats['utilization_deviations'].append(utilization_deviation)
        
        # Логируем событие для этого трамвая
        tram.log_stop_event(
            time=self.env.now,
            stop_id=stop_id,
            direction=tram.direction,
            waiting_before=waiting_before + new_passengers,  # Всего ждали
            alighted=alighted,
            boarded=boarded,
            utilization_after=tram.utilization * 100  # В процентах
        )
        
        # Время стоянки
        boarding_time = (boarded + alighted) * 0.05
        total_stop_time = self.stop_time + boarding_time
        
        yield self.env.timeout(total_stop_time)

    def tram_generator(self):
        """Генератор трамваев - процесс SimPy"""
        while True:
            interval = self.get_current_interval(self.env.now)
            
            # Пытаемся получить доступный трамвай из пула
            tram = yield self.available_trams.get()
            
            # Запускаем трамвай на маршрут
            print(f"[{self.env.now:.1f}] Трамвай #{tram.tram_id} выехал на маршрут "
                  f"(рейс #{tram.stats.total_trips + 1})")
            self.env.process(self.tram_process(tram))
            
            # Ждём до следующего запуска
            yield self.env.timeout(interval)
    
    def create_initial_trams(self, count: int = 20):
        """Создаёт начальный парк трамваев"""
        for _ in range(count):
            self.tram_counter += 1
            tram = Tram(self.tram_counter, self.tram_capacity)
            self.trams[self.tram_counter] = tram
            self.all_trams_created.append(tram)
            self.available_trams.put(tram)
        print(f"Создан парк из {count} трамваев")
    
    def tram_process(self, tram: Tram):
        """Процесс движения трамвая по маршруту"""
        try:
            # Увеличиваем счётчик рейсов в начале
            tram.stats.total_trips += 1
            
            # ОДИН полный цикл: туда и обратно = ОДИН РЕЙС
            for direction in ["forward", "backward"]:
                tram.direction = direction
                
                # Определяем последовательность остановок
                if direction == "forward":
                    stops_sequence = list(range(1, self.stop_number + 1))
                else:
                    stops_sequence = list(range(self.stop_number, 0, -1))
                
                # Проходим все остановки
                for i, stop_id in enumerate(stops_sequence):
                    # Рассчитываем время в пути до этой остановки
                    if i > 0:
                        prev_stop = stops_sequence[i - 1]
                        
                        # Определяем расстояние
                        if direction == "forward":
                            distance = self.distances.get(stop_id, 0)
                        else:
                            distance = self.distances.get(prev_stop, 0)
                        
                        # Едем до остановки
                        hour = int(self.env.now // 60) % 24
                        travel_time = self.calculate_travel_time(distance, hour)
                        yield self.env.timeout(travel_time)
                    
                    # Прибываем на остановку
                    yield self.env.process(self.arrive_at_stop(tram, stop_id))
                
                # Если закончили forward, делаем паузу перед разворотом
                if direction == "forward":
                    yield self.env.timeout(2.0)  # 2 минуты на разворот
            
            # После туда-обратно - трамвай возвращается на базу
            print(f"[{self.env.now:.1f}] Трамвай #{tram.tram_id} вернулся на базу. "
                  f"Рейс #{tram.stats.total_trips}, обслужено {tram.stats.passengers_served} пас. (всего)")
            
            # возвращаем трамвай в пул для повторного использования
            yield self.available_trams.put(tram)
            
        except simpy.Interrupt:
            print(f"[{self.env.now:.1f}] Трамвай #{tram.tram_id} прерван")
    
    def run_simulation(self, plot_graphs: bool = True, save_logs: bool = True):
        """Запуск симуляции"""
        print(f"\n{'='*60}")
        print(f"Запуск симуляции движения трамваев")
        print(f"Конфигурация: {os.path.basename(self.config_file)}")
        print(f"Длительность: {self.simulation_hours} часов")
        print(f"Количество остановок: {self.stop_number}")
        print(f"Вместимость трамвая: {self.tram_capacity} чел.")
        print(f"{'='*60}\n")
        
        self.create_initial_trams(count=8)
        self.env.process(self.tram_generator())
        self.env.run(until=self.simulation_hours * 60)
        self.print_final_stats()
        
        # ИЗМЕНЕНИЕ: передаём пути к папкам
        if save_logs:
            from logger import TramLogger
            logs_dir = os.path.join(self.run_dir, "logs")
            logger = TramLogger(output_dir=logs_dir)
            logger.save_all_trams(self.trams)
            logger.create_summary(self.trams)
        
        
        from visualization import TramVisualization
        plots_dir = os.path.join(self.run_dir, "plots")
        viz = TramVisualization(self.stops, self.simulation_hours)
        viz.create_all_plots(trams=self.trams, output_dir=plots_dir)
        
        print(f"\n{'='*60}")
        print(f"✓ СИМУЛЯЦИЯ ЗАВЕРШЕНА")
        print(f"✓ Результаты сохранены: {self.run_dir}/")
        print(f"{'='*60}\n")

    
    def print_final_stats(self):
        """Вывод финальной статистики"""
        print(f"\n{'='*60}")
        print(f"РЕЗУЛЬТАТЫ СИМУЛЯЦИИ")
        print(f"{'='*60}")
        print(f"Время симуляции: {self.simulation_hours} часов ({self.env.now:.1f} мин)")
        print(f"\nОбщая статистика:")
        print(f"  • Всего обслужено пассажиров: {self.stats['total_passengers_served']}")
        print(f"  • Трамваев в парке: {len(self.all_trams_created)}")
        
        if self.stats['utilization_deviations']:
            avg_util_deviation = sum(self.stats['utilization_deviations']) / len(self.stats['utilization_deviations'])
            print(f"  • Среднее отклонение загруженности от 75%: {avg_util_deviation:.2%}")
        
        # Статистика по трамваям
        print(f"\nСтатистика по трамваям:")
        active_trams = [t for t in self.all_trams_created if t.stats.total_trips > 0]
        active_trams.sort(key=lambda x: x.tram_id)
        
        for tram in active_trams:
            if tram.stats.utilization_history:
                avg_util = sum(tram.stats.utilization_history) / len(tram.stats.utilization_history)
                print(f"  Трамвай #{tram.tram_id}: {tram.stats.passengers_served} пас., "
                      f"{tram.stats.total_trips} рейсов (туда-обратно), средняя загрузка {avg_util:.1%}")
        
        # Статистика по остановкам
        print(f"\nСтатистика по остановкам:")
        for stop_id in sorted(self.stops.keys()):
            stop = self.stops[stop_id]
            if stop.passengers_served > 0:
                avg_waiting = stop.total_waiting_time / stop.passengers_served
                print(f"  Остановка {stop_id:2d}: обслужено {stop.passengers_served:4d} пас., "
                      f"среднее ожидание {avg_waiting:5.1f} мин, "
                      f"осталось {stop.waiting_passengers} в очереди")
        
        print(f"{'='*60}\n")


def main():
    """Главная функция запуска"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Симуляция движения трамваев')
    parser.add_argument('--config', type=str, default='tram_config.json',
                       help='Имя файла конфигурации (из папки configs/)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Не создавать графики')
    parser.add_argument('--no-logs', action='store_true',
                       help='Не сохранять детальные логи трамваев')
    
    args = parser.parse_args()
    
    sim = TramSimulation(args.config)
    sim.run_simulation(
        plot_graphs=not args.no_plots,
        save_logs=not args.no_logs
    )


if __name__ == "__main__":
    main()