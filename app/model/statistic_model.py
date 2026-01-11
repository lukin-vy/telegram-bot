import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from .metrics import Metrics

class Statistic_model():

    def set_data(self, trading_history_df:pd.DataFrame):
        """Загрузка новых данных для прогнозирования"""
        self.__dataset_df = trading_history_df.copy()

    def test(self):
        """Выполнить обучение и тестирование модели"""
        forecast_days = 30
        test_df = self.__dataset_df.copy()

        # 1. Разделение на обучающую и тестовую выборки
        train_df, test_df = self.__holdout_split(
            dataset = test_df,
            test_shape = forecast_days
        )

        # 2. Обучение модели
        self.__learn_model(dataset = train_df)

        # 4. Расчет прогноза
        predict = self.__model.forecast(forecast_days)

        # 5. Расчет метрик
        y_test = test_df['Close'].values
        return Metrics(
            mae = mean_absolute_error(y_test, predict),
            rmse = mean_squared_error(y_test, predict),
            mape = mean_absolute_percentage_error(y_test, predict)
        )

    def fit_predict(self, predict_days=30):
        """Обучить модель и расчитать прогноз"""

        train_df = self.__dataset_df.copy()

        # 1. Обучение модели
        self.__learn_model(train_df)

        # 2.Расчет прогноза
        predict = self.__model.forecast(predict_days)

        # 4. Приведение к историческим данным
        return self.__correct_predict(
            last_price = train_df['Close'].values[-1], 
            predict = predict
        )

    def __holdout_split(self, dataset:pd.DataFrame, test_shape:int):
        """Разделение на обучающую и тестовую выборки"""
        train_df = dataset.iloc[:-test_shape]
        test_df  = dataset.iloc[-test_shape:]
        return train_df, test_df

    def __learn_model(self, dataset:pd.DataFrame):
        """Обучение модели ExponentialSmoothing"""
        self.__model = ExponentialSmoothing(
            dataset['Close'].values,
            trend="add",
            seasonal="add",
            seasonal_periods=120,
            initialization_method="estimated"
        ).fit()

    def __correct_predict(self, last_price, predict):
        "Приведение прогноза к историческим данным"
        delta = abs(predict[0] - last_price)
        if predict[0] > last_price:
            predict -= delta
        else:
            predict += delta
        return predict