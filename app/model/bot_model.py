import numpy as np
import pandas as pd
import yfinance as yf
import io
import matplotlib.pyplot as plt

from datetime import timedelta
from .classic_model import Classic_model
from .statistic_model import Statistic_model
from .deep_model import Deep_model
from . import LOGGER

class Bot_model():

    def __init__(self, ticker):
        self.__ticker = ticker
        self.__trading_history_df = self.__get_stock_data(ticker)
        self.__models = {
            'knn_model': Classic_model(),
            'ets_model': Statistic_model(),
            'lstm_model': Deep_model()
        }

    def predict_price(self):

        metric_list = list()
        # Для всех моделей выполним расчет 
        for key_name in self.__models:
            current_model = self.__models[key_name]
            current_model.set_data(self.__trading_history_df)
            metrics = current_model.test()
            metric_list.append(metrics.rmse)
            LOGGER.to_log(f'Модель {key_name}, показатель RMSE = {metrics.rmse}')
            
        # Получаем название лучшей модели
        best_model_name = list(self.__models.keys())[np.argmin(np.array(metric_list))]
        best_model = self.__models[best_model_name]
        LOGGER.to_log(f'Выбрана модель {best_model_name}')

        # Получем прогноз на 30 дней
        self.__predict = best_model.fit_predict(predict_days=30)

        # Получим график с прогнозом
        return self.__get_chart_image()

    def get_profit(self, invest_summ):

        deals_list = self.__get_deals(self.__predict)
        if not deals_list:
            return

        forecast_dates = self.__get_forecast_dates()
            
        balanse = invest_summ
        positions = 0
        deals_text = f"Началная сумма {invest_summ}$\nСписок сделок:\n"
        LOGGER.to_log(f'Начальная сумма инвестиций {invest_summ}$')

        for index, action, price in deals_list:

            action_date = forecast_dates[index].strftime("%d.%m.%Y")

            if action == 'buy':
                # Получаем акции
                positions = balanse // price
                # Списываем деньги
                balanse -= round(positions * price, 2)

                deals_text += f'{action_date} покупка {positions} шт. по цене {price:.2f}$\n'

            if action == 'sell':
                deals_text += f'{action_date} продажа {positions} шт. по цене {price:.2f}$\n'
                
                # Зачисляем деньги
                balanse += round(positions * price, 2)
                # Списываем акции
                positions = 0

        profit = (balanse - invest_summ) / invest_summ * 100
        deals_text += f'\nИтого:\n'
        deals_text += f'Сумма портфеля: {balanse:.2f}$\n'
        deals_text += f'Прирост: {round(profit,2)}%'

        profit_chart = self.__get_chart_image(
            deals = deals_list,
            last_period = 30
        )

        LOGGER.to_log(f'Конечная сумма {balanse:.2f}$')
        LOGGER.to_log(f'Доход {round(balanse - invest_summ,2)}$')
        LOGGER.to_log(f'Доходность {round(profit,2)}%')

        return deals_text, profit_chart

    def __get_stock_data(self, ticker:str)->pd.DataFrame:
        """Получить исторические сведения о стоимости акции"""

        if len(ticker) == 0:
            raise ValueError('Название тикера отсутствует!')

        ticker.upper()
        stock = yf.Ticker(ticker)
        trading_history_df = stock.history(period="2y", interval="1d")

        if trading_history_df.empty:
            raise ValueError(f'Данные по торгам компании {ticker} не найлены.')
        else:
            trading_history_df.drop(['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits'], axis=1, inplace=True)
            trading_history_df.sort_index(inplace=True)
            return trading_history_df

    def __get_forecast_dates(self):
        # Создаем даты для прогноза (начиная со дня после последней известной даты)
        last_date = self.__trading_history_df.index[-1]
        return [last_date + timedelta(days=i) for i in range(1, len(self.__predict) + 1)]

    def __get_deals(self, prices, threshold_percent=0.2):
        if len(prices) < 2:
            return []
        
        n = len(prices)
        extremums = []
        
        # Шаг 1: Находим все локальные экстремумы
        for i in range(n):
            if i == 0:
                if prices[i] < prices[i + 1]:
                    extremums.append((i, 'min', prices[i]))
                elif prices[i] > prices[i + 1]:
                    extremums.append((i, 'max', prices[i]))
            elif i == n - 1:
                if prices[i] < prices[i - 1]:
                    extremums.append((i, 'min', prices[i]))
                elif prices[i] > prices[i - 1]:
                    extremums.append((i, 'max', prices[i]))
            else:
                # Локальный минимум
                if prices[i] <= prices[i - 1] and prices[i] <= prices[i + 1]:
                    # Проверяем, что это не плато (ровный участок)
                    if not (prices[i] == prices[i - 1] and prices[i] == prices[i + 1]):
                        extremums.append((i, 'min', prices[i]))
                # Локальный максимум
                elif prices[i] >= prices[i - 1] and prices[i] >= prices[i + 1]:
                    # Проверяем, что это не плато (ровный участок)
                    if not (prices[i] == prices[i - 1] and prices[i] == prices[i + 1]):
                        extremums.append((i, 'max', prices[i]))
        
        # Шаг 2: Объединяем соседние экстремумы одного типа
        filtered_extremums = []
        i = 0
        while i < len(extremums):
            current_idx, current_type, current_price = extremums[i]
            
            # Объединяем несколько соседних минимумов/максимумов в один
            j = i + 1
            best_idx, best_price = current_idx, current_price
            
            while j < len(extremums) and extremums[j][1] == current_type:
                j_idx, j_type, j_price = extremums[j]
                
                if current_type == 'min' and j_price < best_price:
                    best_idx, best_price = j_idx, j_price
                elif current_type == 'max' and j_price > best_price:
                    best_idx, best_price = j_idx, j_price
                j += 1
            
            filtered_extremums.append((best_idx, current_type, best_price))
            i = j
        
        # Шаг 3: Начинаем с покупки и чередуем покупки/продажи
        result = []
        
        # Ищем первую покупку
        start_idx = 0
        while start_idx < len(filtered_extremums) and filtered_extremums[start_idx][1] != 'min':
            start_idx += 1
        
        if start_idx >= len(filtered_extremums):
            return []  # Нет подходящей точки для покупки
        
        # Добавляем первую покупку
        buy_idx, _, buy_price = filtered_extremums[start_idx]
        result.append((buy_idx, 'buy', buy_price))
        
        # Проходим по оставшимся экстремумам
        i = start_idx + 1
        last_action = 'buy'
        last_price = buy_price
        
        while i < len(filtered_extremums):
            current_idx, current_type, current_price = filtered_extremums[i]
            
            # Рассчитываем процент изменения
            price_change = 0.0
            if last_price > 0:
                if last_action == 'buy' and current_type == 'max':
                    price_change = ((current_price - last_price) / last_price) * 100
                elif last_action == 'sell' and current_type == 'min':
                    price_change = ((last_price - current_price) / last_price) * 100
            
            # Проверяем, нужно ли совершать сделку
            should_trade = False
            
            if last_action == 'buy' and current_type == 'max':
                # Продажа после покупки
                should_trade = price_change >= threshold_percent
            elif last_action == 'sell' and current_type == 'min':
                # Покупка после продажи
                should_trade = price_change >= threshold_percent
            
            # Также совершаем сделку при смене тренда, даже если не достигнут порог
            
            if should_trade:
                action = 'sell' if last_action == 'buy' else 'buy'
                result.append((current_idx, action, current_price))
                last_action = action
                last_price = current_price
            
            i += 1
        
        # Шаг 4: Очищаем результат - удаляем последнюю сделку, если это покупка без последующей продажи
        if len(result) > 0 and result[-1][1] == 'buy':
            result = result[:-1]
        
        return result

    def __get_chart_image(self, deals = list(), last_period = 0):
        """Сформировать график с прогнозом цены"""
        plt.figure(figsize=(16, 8))

        if last_period == 0:
            actual_dates = self.__trading_history_df.index
            actual_prices = self.__trading_history_df['Close'].values
        else:
            actual_dates = self.__trading_history_df.iloc[-last_period:].index
            actual_prices = self.__trading_history_df.iloc[-last_period:]['Close'].values
        
        forecast_dates = self.__get_forecast_dates()
        
        # Фактические цены
        plt.plot(actual_dates, actual_prices, 'b-', linewidth=2, label='Исторические цены')
        
        # Прогноз
        plt.plot(forecast_dates, self.__predict, 'r--', linewidth=2, label='Прогноз')
        
        # Разделительная вертикальная линия
        last_date = self.__trading_history_df.index[-1]
        plt.axvline(x=last_date, color='gray', linestyle='--', alpha=0.7, linewidth=1)

        for index, action, price in deals:
            action_date = forecast_dates[index]
            # Отметка о покупке
            if action == 'buy':
                plt.scatter([action_date], [price], color='green', s=100, zorder=5,
                            label=f'Покупка: {price:.2f}')
                plt.annotate(f'Покупка {price:.2f}', 
                             xy=(action_date, price),
                             xytext=(0, -15),
                             textcoords='offset points',
                             ha='center',
                             fontsize=10,
                             color='black')
            
            # Отметка о продаже
            if action == 'sell':
                plt.scatter([action_date], [price], color='red', s=100, zorder=5,
                            label=f'Продажа: {price:.2f}')
                plt.annotate(f'Продажа {price:.2f}', 
                             xy=(action_date, price),
                             xytext=(0, 10),
                             textcoords='offset points',
                             ha='center',
                             fontsize=10,
                             color='black')            
        
         # Настройки графика
        plt.title(f'Прогноз цен акций {self.__ticker} на 30 дней', 
                  fontsize=16)
        plt.xlabel('Дата', fontsize=12)
        plt.ylabel('Цена ($)', fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(loc='best', fontsize=10)

        global_min = min(np.min(actual_prices), np.min(self.__predict))
        
        # Добавляем аннотации
        plt.text(
            last_date - timedelta(days=len(actual_dates)//1.5),
            global_min, 
            'ИСТОРИЧЕСКИЕ ДАННЫЕ',
            ha='center',
            fontsize=10,
            color='blue'
        )
        
        plt.text(
            last_date + timedelta(days=len(self.__predict)//1.5),
            global_min, 
            'ПРОГНОЗ',
            ha='center',
            fontsize=10,
            color='red'
        )
        
        # Добавляем подписи дат для прогноза
        forecast_start = forecast_dates[0]
        forecast_mid = forecast_dates[30//2]
        forecast_end = forecast_dates[-1]
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=300, bbox_inches='tight')
        img_buf.seek(0)
        plt.close()

        return img_buf