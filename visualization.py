"""
Модуль визуализации результатов симуляции трамваев
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from typing import Dict, List, Tuple
import os

matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


class TramVisualization:
    """Класс для визуализации результатов симуляции"""
    
    def __init__(self, stops: Dict, simulation_hours: int = 24):
        self.stops = stops
        self.stop_number = len(stops)
        self.simulation_hours = simulation_hours
    
    def plot_waiting_passengers(self, output_file: str = "waiting_passengers.png"):
        print(f"\nСоздание графика динамики очередей...")
        
        # Определяем количество строк и столбцов
        ncols = 3
        nrows = (self.stop_number + ncols - 1) // ncols
        
        # Создаём фигуру
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 4))
        fig.suptitle('Количество ожидающих пассажиров на остановках', 
                     fontsize=16, fontweight='bold')
        
        # Преобразуем оси в плоский список
        if nrows == 1:
            axes_flat = axes if self.stop_number > 1 else [axes]
        else:
            axes_flat = axes.flatten()
        
        # Строим график для каждой остановки
        for stop_id in range(1, self.stop_number + 1):
            stop = self.stops[stop_id]
            ax = axes_flat[stop_id - 1]
            
            if not stop.waiting_history:
                ax.text(0.5, 0.5, 'Нет данных', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Остановка {stop_id}')
                continue
            
            # Извлекаем данные
            times = [t / 60 for t, _ in stop.waiting_history]  # Часы
            counts = [c for _, c in stop.waiting_history]
            
            # График
            ax.plot(times, counts, linewidth=1.5, color='steelblue', alpha=0.7)
            ax.fill_between(times, counts, alpha=0.3, color='lightblue')
            
            # Оформление
            ax.set_title(f'Остановка {stop_id}', fontweight='bold')
            ax.set_xlabel('Время (часы)', fontsize=9)
            ax.set_ylabel('Количество пассажиров', fontsize=9)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlim(0, self.simulation_hours)
            
            # Статистика
            max_waiting = max(counts) if counts else 0
            avg_waiting = sum(counts) / len(counts) if counts else 0
            ax.text(0.98, 0.95, f'Max: {max_waiting}\nAvg: {avg_waiting:.0f}',
                   transform=ax.transAxes, fontsize=8,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Убираем лишние подграфики
        for idx in range(self.stop_number, len(axes_flat)):
            fig.delaxes(axes_flat[idx])
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"График сохранён: {output_file}")
    
    def plot_waiting_by_hour(self, output_file: str = "waiting_by_hour.png",
                            plot_all: bool = False):
        """
        Создаёт график среднего количества ожидающих пассажиров
        по часам для каждой остановки
        """
        print(f"\nСоздание графика по часам...")
        
        # Подготовка данных: группируем по часам
        hours = list(range(24))
        stop_data = {}
        
        for stop_id in range(1, self.stop_number + 1):
            stop = self.stops[stop_id]
            hourly_counts = [[] for _ in range(24)]
            
            for time, count in stop.waiting_history:
                hour = int(time // 60) % 24
                hourly_counts[hour].append(count)
            
            # Вычисляем средние значения
            stop_data[stop_id] = [
                np.mean(counts) if counts else 0 
                for counts in hourly_counts
            ]
        
        # Создаём график
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Выбираем остановки для отображения
        if plot_all:
            selected_stops = list(range(1, self.stop_number + 1))
        else:
            selected_stops = list(range(1, self.stop_number + 1, 2))
        
        # Цветовая палитра
        colors = plt.cm.tab20(np.linspace(0, 1, len(selected_stops)))
        
        for idx, stop_id in enumerate(selected_stops):
            ax.plot(hours, stop_data[stop_id], 
                   marker='o', linewidth=2, markersize=4,
                   color=colors[idx],
                   label=f'Остановка {stop_id}')
        
        # Оформление
        ax.set_title('Среднее количество ожидающих пассажиров по часам', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Час дня', fontsize=12)
        ax.set_ylabel('Среднее количество пассажиров', fontsize=12)
        ax.set_xticks(hours)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=9, ncol=2)
        
        # Выделяем часы пик
        ax.axvspan(7, 9, alpha=0.1, color='red')
        ax.axvspan(17, 19, alpha=0.1, color='orange')
        
        # Подписи часов пик
        ax.text(8, ax.get_ylim()[1] * 0.95, 'Утренний\nчас пик',
               ha='center', fontsize=9, alpha=0.7)
        ax.text(18, ax.get_ylim()[1] * 0.95, 'Вечерний\nчас пик',
               ha='center', fontsize=9, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"График сохранён: {output_file}")
    
    def plot_heatmap(self, output_file: str = "waiting_heatmap.png"):
        """
        Создаёт тепловую карту количества ожидающих пассажиров
        (остановка × час)
        """
        print(f"\nСоздание тепловой карты...")
        
        # Подготовка данных
        hours = list(range(24))
        stops = list(range(1, self.stop_number + 1))
        data = np.zeros((self.stop_number, 24))
        
        for stop_id in stops:
            stop = self.stops[stop_id]
            hourly_counts = [[] for _ in range(24)]
            
            for time, count in stop.waiting_history:
                hour = int(time // 60) % 24
                hourly_counts[hour].append(count)
            
            for hour in range(24):
                if hourly_counts[hour]:
                    data[stop_id - 1, hour] = np.mean(hourly_counts[hour])
        
        # Создаём тепловую карту
        fig, ax = plt.subplots(figsize=(14, max(8, self.stop_number * 0.4)))
        
        im = ax.imshow(data, cmap='YlOrRd', aspect='auto', interpolation='nearest')
        
        # Оси
        ax.set_xticks(range(24))
        ax.set_xticklabels(hours)
        ax.set_yticks(range(self.stop_number))
        ax.set_yticklabels([f'Ост. {i}' for i in stops])
        
        # Названия
        ax.set_title('Тепловая карта: количество ожидающих пассажиров', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Час дня', fontsize=12)
        ax.set_ylabel('Остановка', fontsize=12)
        
        # Цветовая шкала
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Количество пассажиров', fontsize=10)
        
        # Сетка
        ax.set_xticks(np.arange(24) - 0.5, minor=True)
        ax.set_yticks(np.arange(self.stop_number) - 0.5, minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Тепловая карта сохранена: {output_file}")
    
    def plot_utilization(self, trams: Dict, output_file: str = "tram_utilization.png"):
        """
        Создаёт график загруженности трамваев
        """
        print(f"\nСоздание графика загруженности трамваев...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # График 1: Средняя загруженность по трамваям
        active_trams = [t for t in trams.values() if t.stats.total_trips > 0]
        active_trams.sort(key=lambda x: x.tram_id)
        
        tram_ids = [t.tram_id for t in active_trams]
        avg_utils = []
        
        for tram in active_trams:
            if tram.stats.utilization_history:
                avg_util = np.mean(tram.stats.utilization_history) * 100
                avg_utils.append(avg_util)
            else:
                avg_utils.append(0)
        
        colors = ['green' if 60 <= u <= 80 else 'orange' if u > 80 else 'red' 
                 for u in avg_utils]
        
        ax1.bar(tram_ids, avg_utils, color=colors, alpha=0.7, edgecolor='black')
        ax1.axhline(y=75, color='blue', linestyle='--', linewidth=2, label='Целевая (75%)')
        ax1.axhline(y=60, color='green', linestyle=':', alpha=0.5, label='Комфортная зона')
        ax1.axhline(y=90, color='red', linestyle=':', alpha=0.5, label='Перегруз')
        
        ax1.set_title('Средняя загруженность трамваев', fontweight='bold')
        ax1.set_xlabel('ID трамвая')
        ax1.set_ylabel('Загруженность (%)')
        ax1.set_ylim(0, 110)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # График 2: Количество рейсов
        trips = [t.stats.total_trips for t in active_trams]
        passengers = [t.stats.passengers_served for t in active_trams]
        
        ax2.bar(tram_ids, trips, alpha=0.7, label='Рейсы', color='steelblue')
        ax2_twin = ax2.twinx()
        ax2_twin.plot(tram_ids, passengers, color='red', marker='o', 
                     linewidth=2, markersize=6, label='Пассажиры')
        
        ax2.set_title('Рейсы и обслуженные пассажиры', fontweight='bold')
        ax2.set_xlabel('ID трамвая')
        ax2.set_ylabel('Количество рейсов', color='steelblue')
        ax2_twin.set_ylabel('Обслужено пассажиров', color='red')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"График загруженности сохранён: {output_file}")
    
    def plot_tram_utilization_by_hour(self, trams: Dict, output_file: str = "tram_utilization_by_hour.png"):
        """
        Создаёт график средней загруженности трамваев по часам
        """
        print(f"\nСоздание графика загруженности по часам...")
        
        # Собираем данные: для каждого трамвая группируем загруженность по часам
        tram_hourly_data = {}
        
        for tram_id, tram in trams.items():
            if not tram.stats.stop_log:
                continue
            
            hourly_utils = [[] for _ in range(24)]
            
            for event in tram.stats.stop_log:
                hour = int(event['time'] // 60) % 24
                hourly_utils[hour].append(event['utilization_after'])
            
            # Вычисляем средние значения
            tram_hourly_data[tram_id] = [
                np.mean(utils) if utils else 0 
                for utils in hourly_utils
            ]
        
        # Вычисляем общую среднюю загруженность по часам (все трамваи)
        hours = list(range(24))
        avg_utilization = []
        
        for hour in hours:
            hour_values = [
                tram_hourly_data[tram_id][hour] 
                for tram_id in tram_hourly_data 
                if tram_hourly_data[tram_id][hour] > 0
            ]
            avg_utilization.append(np.mean(hour_values) if hour_values else 0)
        
        # Создаём график
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # ГРАФИК 1: Средняя загруженность всех трамваев по часам
        ax1.plot(hours, avg_utilization, linewidth=3, color='steelblue', 
                marker='o', markersize=6, label='Средняя загруженность')
        ax1.fill_between(hours, avg_utilization, alpha=0.3, color='lightblue')
        
        # Целевая линия 75%
        ax1.axhline(y=75, color='green', linestyle='--', linewidth=2, 
                    label='Целевая загруженность (75%)', alpha=0.7)
        
        # Комфортная зона 60-80%
        ax1.axhspan(60, 80, alpha=0.1, color='green', label='Комфортная зона')
        
        # Выделяем часы пик
        ax1.axvspan(7, 9, alpha=0.1, color='red')
        ax1.axvspan(17, 19, alpha=0.1, color='orange')
        
        ax1.set_title('Средняя загруженность трамваев по часам суток', 
                    fontsize=14, fontweight='bold')
        ax1.set_xlabel('Час дня', fontsize=12)
        ax1.set_ylabel('Загруженность (%)', fontsize=12)
        ax1.set_xticks(hours)
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='best', fontsize=10)
        
        # Подписи часов пик
        ax1.text(8, max(avg_utilization) * 1.05 if max(avg_utilization) > 0 else 10, 
                'Утренний\nчас пик', ha='center', fontsize=9, alpha=0.7)
        ax1.text(18, max(avg_utilization) * 1.05 if max(avg_utilization) > 0 else 10, 
                'Вечерний\nчас пик', ha='center', fontsize=9, alpha=0.7)
        
        # ГРАФИК 2: Загруженность каждого трамвая (выборочно - каждый второй)
        selected_trams = sorted(list(tram_hourly_data.keys()))[::2]  # Каждый второй
        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_trams)))
        
        for idx, tram_id in enumerate(selected_trams):
            ax2.plot(hours, tram_hourly_data[tram_id], 
                    linewidth=1.5, marker='o', markersize=3,
                    color=colors[idx], alpha=0.7,
                    label=f'Трамвай #{tram_id}')
        
        # Средняя линия для референса
        ax2.plot(hours, avg_utilization, linewidth=2.5, color='black', 
                linestyle='--', label='Средняя', alpha=0.5)
        
        ax2.set_title('Загруженность отдельных трамваев по часам', 
                    fontsize=14, fontweight='bold')
        ax2.set_xlabel('Час дня', fontsize=12)
        ax2.set_ylabel('Загруженность (%)', fontsize=12)
        ax2.set_xticks(hours)
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(loc='best', fontsize=9, ncol=2)
        
        # Выделяем часы пик
        ax2.axvspan(7, 9, alpha=0.1, color='red')
        ax2.axvspan(17, 19, alpha=0.1, color='orange')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"График загруженности по часам сохранён: {output_file}")
        
        # Выводим статистику
        print(f"Статистика загруженности:")
        print(f"    Средняя за сутки: {np.mean(avg_utilization):.1f}%")
        print(f"    Максимальная: {max(avg_utilization):.1f}% в {avg_utilization.index(max(avg_utilization))}:00")
        print(f"    Минимальная: {min(avg_utilization):.1f}% в {avg_utilization.index(min(avg_utilization))}:00")
        
        # Статистика по часам пик
        morning_peak = avg_utilization[7:10]  # 7-9
        evening_peak = avg_utilization[17:20]  # 17-19
        if morning_peak:
            print(f"    Утренний час пик (7-9): {np.mean(morning_peak):.1f}%")
        if evening_peak:
            print(f"    Вечерний час пик (17-19): {np.mean(evening_peak):.1f}%")

    def create_all_plots(self, trams: Dict = None, output_dir: str = "."):
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("СОЗДАНИЕ ГРАФИКОВ")
        print(f"Папка: {output_dir}/")
        print("="*60)
        
        try:
            self.plot_waiting_passengers(f"{output_dir}/waiting_passengers.png")
            self.plot_waiting_by_hour(f"{output_dir}/waiting_by_hour.png")
            self.plot_heatmap(f"{output_dir}/waiting_heatmap.png")
            
            if trams:
                self.plot_utilization(trams, f"{output_dir}/tram_utilization.png")
                self.plot_tram_utilization_by_hour(trams, f"{output_dir}/tram_utilization_by_hour.png")
            
            print("\n" + "="*60)
            print("ВСЕ ГРАФИКИ УСПЕШНО СОЗДАНЫ!")
            print("="*60)
            
        except Exception as e:
            print(f"\nОшибка при создании графиков: {e}")
            import traceback
            traceback.print_exc()
