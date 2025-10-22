"""
Генератор JSON конфигураций для имитационной модели трамваев
"""

import json
from typing import List, Tuple
import os
CONFIG_DIR = "configs"


def ensure_config_dir():
    """Создаёт папку для конфигураций, если её нет"""
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR)
        print(f"Создана папка для конфигураций: {CONFIG_DIR}/")

def generate_intensity_data(stop_number: int, 
                            base_intensities: List[int] = None) -> List[List[int]]:
    """Генерирует данные об интенсивности прибытия пассажиров"""
    intensity_data = []
    
    if base_intensities is None:
        base_intensities = []
        for i in range(1, stop_number + 1):
            # УВЕЛИЧИВАЕМ базовую интенсивность в 3-4 раза
            center = stop_number / 2
            distance_from_center = abs(i - center)
            base = 80 + int((stop_number - distance_from_center) * 5)  # было 100 + 10
            base_intensities.append(base)
    
    time_coefficients = {
        0: 0.02, 1: 0.01, 2: 0.01, 3: 0.01, 4: 0.02, 5: 0.1,
        6: 0.8,   # Утро
        7: 2.0,   # Час пик - УВЕЛИЧЕНО
        8: 2.5,   # Час пик - УВЕЛИЧЕНО
        9: 1.2,   # После пика
        10: 0.9, 11: 0.9, 12: 1.0, 13: 0.9, 14: 0.8, 15: 0.9, 16: 1.2,
        17: 2.2,  # Час пик - УВЕЛИЧЕНО
        18: 2.8,  # Час пик - УВЕЛИЧЕНО
        19: 1.5,  # После пика
        20: 0.8, 21: 0.5, 22: 0.2, 23: 0.05
    }
    
    for stop_id in range(1, stop_number + 1):
        base = base_intensities[stop_id - 1]
        for hour in range(24):
            coef = time_coefficients.get(hour, 0.5)
            intensity = int(base * coef)
            intensity_data.append([stop_id, hour, intensity])
    
    return intensity_data


def generate_bus_intervals(peak_hours: List[Tuple[int, int]] = None) -> List[List[int]]:
    """Генерирует интервалы выхода трамваев"""
    if peak_hours is None:
        # УВЕЛИЧИВАЕМ интервалы - меньше трамваев
        return [
            [0, 120],  # Ночь - каждые 2 часа (было 60)
            [6, 30],   # Утро - каждые 30 минут (было 20)
            [8, 15],   # Час пик - каждые 15 минут (было 8)
            [10, 20],  # День - каждые 20 минут (было 15)
            [17, 15],  # Вечерний час пик - каждые 15 минут (было 8)
            [20, 30],  # Вечер - каждые 30 минут (было 20)
            [23, 120]  # Ночь - каждые 2 часа (было 60)
        ]
    else:
        return peak_hours


def generate_road_loads(peak_hours: List[int] = None) -> List[List[float]]:
    """
    Генерирует данные о загруженности дорог
    
    Args:
        peak_hours: список часов с максимальной загруженностью
    
    Returns:
        List of [hour, load_factor (0-1)]
    """
    if peak_hours is None:
        peak_hours = [8, 9, 17, 18]  # Часы пик по умолчанию
    
    road_loads = []
    for hour in range(24):
        if hour in peak_hours:
            load = 0.9  # Высокая загруженность
        elif hour in range(7, 10) or hour in range(16, 20):
            load = 0.7  # Средне-высокая
        elif hour in range(10, 16):
            load = 0.6  # Средняя (дневное время)
        elif hour in range(20, 23) or hour == 6:
            load = 0.4  # Низкая
        else:
            load = 0.1  # Очень низкая (ночь)
        
        road_loads.append([hour, load])
    
    return road_loads


def generate_distances(stop_number: int, 
                      min_distance: int = 400,
                      max_distance: int = 800) -> List[List[int]]:
    """
    Генерирует расстояния между остановками
    
    Args:
        stop_number: количество остановок
        min_distance: минимальное расстояние в метрах
        max_distance: максимальное расстояние в метрах
    
    Returns:
        List of [stop_id, distance_from_previous]
    """
    import random
    
    distances = [[1, 0]]  # Первая остановка - начальная точка
    
    for i in range(2, stop_number + 1):
        # Генерируем случайное расстояние
        distance = random.randint(min_distance, max_distance)
        # Округляем до 50 метров
        distance = (distance // 50) * 50
        distances.append([i, distance])
    
    return distances

def save_config_compact(config: dict, output_file: str):
    """
    Сохраняет конфигурацию в компактном читаемом формате
    """
    # Убедимся, что папка существует
    ensure_config_dir()
    
    # ИЗМЕНЕНИЕ: сохраняем в папку configs/
    if not output_file.startswith(CONFIG_DIR):
        output_file = os.path.join(CONFIG_DIR, os.path.basename(output_file))
    
    lines = ["{"]
    
    # Простые поля
    lines.append(f'  "stop_number": {config["stop_number"]},')
    
    # Distance - вся строка в одну линию
    distance_str = json.dumps(config["distance"], ensure_ascii=False)
    lines.append(f'  "distance": {distance_str},')
    
    # Intensity - по строчке на остановку
    lines.append('  "intensity": [')
    stop_number = config["stop_number"]
    
    for stop_id in range(1, stop_number + 1):
        stop_data = [item for item in config["intensity"] if item[0] == stop_id]
        stop_items = ", ".join([json.dumps(item) for item in stop_data])
        comma = "," if stop_id < stop_number else ""
        lines.append(f'    {stop_items}{comma}')
    
    lines.append('  ],')
    
    # Bus intervals - вся строка в одну линию
    bus_interval_str = json.dumps(config["bus_interval"], ensure_ascii=False)
    lines.append(f'  "bus_interval": {bus_interval_str},')
    
    # Road loads - вся строка в одну линию
    road_loads_str = json.dumps(config["road_loads"], ensure_ascii=False)
    lines.append(f'  "road_loads": {road_loads_str},')
    
    # Остальные поля
    lines.append(f'  "flow_speed": {config["flow_speed"]},')
    lines.append(f'  "peak_stop": {config["peak_stop"]},')
    lines.append(f'  "tram_capacity": {config["tram_capacity"]},')
    lines.append(f'  "simulation_hours": {config["simulation_hours"]},')
    lines.append(f'  "acceleration_time": {config["acceleration_time"]},')
    lines.append(f'  "stop_time": {config["stop_time"]}')
    
    lines.append("}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"Конфигурация сохранена: {output_file}")


# В generate_config.py добавь проверку после генерации
def create_config(stop_number: int = 10,
                 tram_capacity: int = 150,
                 simulation_hours: int = 24,
                 flow_speed: int = 40,
                 peak_stop: int = None,
                 output_file: str = "tram_config.json") -> dict:
    """
    Создаёт полную конфигурацию и сохраняет в JSON
    """
    if peak_stop is None:
        peak_stop = (stop_number + 1) // 2
    
    config = {
        "stop_number": stop_number,
        "distance": generate_distances(stop_number),
        "intensity": generate_intensity_data(stop_number),
        "bus_interval": generate_bus_intervals(),
        "road_loads": generate_road_loads(),
        "flow_speed": flow_speed,
        "peak_stop": peak_stop,
        "tram_capacity": tram_capacity,
        "simulation_hours": simulation_hours,
        "acceleration_time": 0.5,
        "stop_time": 1.0
    }
    
    # ИСПОЛЬЗУЕМ НОВУЮ ФУНКЦИЮ
    save_config_compact(config, output_file)
    
    print(f"  • Остановок: {stop_number}")
    print(f"  • Вместимость трамвая: {tram_capacity} чел.")
    print(f"  • Длительность симуляции: {simulation_hours} ч.")
    print(f"  • Записей интенсивности: {len(config['intensity'])}")
    
    return config



def create_custom_config(stop_number: int,
                        base_intensities: List[int],
                        distances: List[int],
                        peak_stop: int,
                        output_file: str = "custom_config.json"):
    """
    Создаёт кастомную конфигурацию с заданными параметрами
    
    Args:
        stop_number: количество остановок
        base_intensities: список базовых интенсивностей для каждой остановки
        distances: список расстояний между остановками (в метрах)
        peak_stop: номер самой популярной остановки
        output_file: имя выходного файла
    """
    # Преобразуем distances в нужный формат
    distance_data = [[1, 0]]  # Первая остановка
    for i, dist in enumerate(distances, start=2):
        distance_data.append([i, dist])
    
    config = {
        "stop_number": stop_number,
        "distance": distance_data,
        "intensity": generate_intensity_data(stop_number, base_intensities),
        "bus_interval": generate_bus_intervals(),
        "road_loads": generate_road_loads(),
        "flow_speed": 40,
        "peak_stop": peak_stop,
        "tram_capacity": 150,
        "simulation_hours": 24,
        "acceleration_time": 0.5,
        "stop_time": 1.0
    }
    
    save_config_compact(config, output_file)
    
    print(f"Кастомная конфигурация сохранена в {output_file}")
    
    return config

# Примеры использования
if __name__ == "__main__":
    print("="*60)
    print("Генератор конфигураций для модели трамваев")
    print("="*60)
    print()
    
    # Создаём папку
    ensure_config_dir()
    
    # Простая конфигурация
    print("1. Создаём простую тестовую конфигурацию...")
    create_config(
        stop_number=5,
        tram_capacity=120,
        simulation_hours=2,
        output_file="tram_config_simple.json"
    )
    print()
    
    # Стандартная конфигурация
    print("2. Создаём стандартную конфигурацию...")
    create_config(
        stop_number=10,
        tram_capacity=120,
        simulation_hours=24,
        output_file="tram_config.json"
    )
    print()
    
    # Большая конфигурация
    print("3. Создаём расширенную конфигурацию...")
    create_config(
        stop_number=15,
        tram_capacity=120,
        simulation_hours=24,
        output_file="tram_config_large.json"
    )
    print()
    
    print("="*60)
    print(f"✓ Все конфигурации сохранены в папку: {CONFIG_DIR}/")
    print("="*60)