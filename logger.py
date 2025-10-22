"""
Модуль логирования детальных событий трамваев
"""

import os
import csv
from typing import Dict, List


class TramLogger:
    """Класс для сохранения логов трамваев"""
    
    def __init__(self, output_dir: str = "tram_logs"):
        self.output_dir = output_dir
        self._create_output_dir()
    
    def _create_output_dir(self):
        """Создаёт директорию для логов, если её нет"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Создана папка для логов: {self.output_dir}/")
    
    def save_tram_log(self, tram_id: int, stop_log: List[dict]):
        """
        Сохраняет лог остановок для одного трамвая
        """
        if not stop_log:
            print(f"Трамвай #{tram_id}: нет событий для записи")
            return
        
        # Формируем имя файла с ведущими нулями
        filename = f"tram_{tram_id:03d}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        # Записываем CSV
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'Время (мин)',
                'Час',
                'Остановка',
                'Направление',
                'Ожидало на остановке',
                'Высадка',
                'Посадка',
                'Загруженность (%)'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for event in stop_log:
                writer.writerow({
                    'Время (мин)': f"{event['time']:.2f}",
                    'Час': int(event['time'] // 60) % 24,
                    'Остановка': event['stop_id'],
                    'Направление': 'Туда' if event['direction'] == 'forward' else 'Обратно',
                    'Ожидало на остановке': event['waiting_before'],
                    'Высадка': event['alighted'],
                    'Посадка': event['boarded'],
                    'Загруженность (%)': f"{event['utilization_after']:.1f}"
                })
        
        print(f"Лог трамвая #{tram_id}: {len(stop_log)} событий → {filename}")
    
    def save_all_trams(self, trams: Dict):
        """
        Сохраняет логи всех трамваев
        """
        print(f"\n{'='*60}")
        print(f"СОХРАНЕНИЕ ЛОГОВ ТРАМВАЕВ")
        print(f"{'='*60}")
        
        saved_count = 0
        for tram_id, tram in trams.items():
            if tram.stats.stop_log:
                self.save_tram_log(tram_id, tram.stats.stop_log)
                saved_count += 1
        
        print(f"\n{'='*60}")
        print(f"Сохранено логов: {saved_count} из {len(trams)} трамваев")
        print(f"Папка: {self.output_dir}/")
        print(f"{'='*60}")
    
    def create_summary(self, trams: Dict, output_file: str = "trams_summary.csv"):
        """
        Создаёт сводную таблицу по всем трамваям
        """
        filepath = os.path.join(self.output_dir, output_file)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'ID трамвая',
                'Всего рейсов',
                'Обслужено пассажиров',
                'Средняя загруженность (%)',
                'Макс загруженность (%)',
                'Всего остановок посещено'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for tram_id, tram in sorted(trams.items()):
                if not tram.stats.utilization_history:
                    continue
                
                avg_util = sum(tram.stats.utilization_history) / len(tram.stats.utilization_history)
                max_util = max(tram.stats.utilization_history)
                
                writer.writerow({
                    'ID трамвая': tram_id,
                    'Всего рейсов': tram.stats.total_trips,
                    'Обслужено пассажиров': tram.stats.passengers_served,
                    'Средняя загруженность (%)': f"{avg_util * 100:.1f}",
                    'Макс загруженность (%)': f"{max_util * 100:.1f}",
                    'Всего остановок посещено': len(tram.stats.stop_log)
                })
        
        print(f"Сводная таблица: {output_file}")
