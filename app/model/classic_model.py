import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from .metrics import Metrics

class Classic_model():

    def __init__(self):
        self.__model = model = KNeighborsRegressor(
            n_neighbors=30,
            weights='uniform',
            metric='euclidean',
            algorithm='auto'
        )

    def set_data(self, trading_history_df:pd.DataFrame):
        """Загрузка новых данных для прогнозирования"""
        self.__dataset_df = trading_history_df.copy()

    def test(self):
        """Выполнить обучение и тестирование модели"""
        forecast_days = 30
        test_df = self.__dataset_df.copy()
        
        # 1. Добавление признаков (временные лаги)
        test_df = self.__add_features(
            dataset = test_df,
            lag_days = forecast_days
        )

        # 2. Разделение на обучающую и тестовую выборки
        train_df, test_df = self.__holdout_split(
            dataset = test_df,
            test_shape = forecast_days
        )

        # 3. Обучение модели
        self.__learn_model(dataset = train_df)

        # 4. Расчет прогноза
        features = train_df['Close'].values[-1:-(forecast_days+1):-1].copy()
        predict = self.__predict(
            predict_days = forecast_days,
            features = features
        )

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

        # 1. Добавление признаков (временные лаги)
        train_df = self.__add_features(
            dataset = train_df,
            lag_days = predict_days
        )

        # 2. Обучение модели
        self.__learn_model(train_df)

        # 3.Расчет прогноза
        features = train_df['Close'].values[-1:-(predict_days+1):-1].copy()
        predict = self.__predict(
            predict_days = predict_days,
            features = features
        )

        # 4. Приведение к историческим данным
        return self.__correct_predict(
            last_price = train_df['Close'].values[-1], 
            predict = predict
        )

    def __predict(self, features, predict_days=30):
        """Расчитать прогноз"""
        predictions = list()
        
        for i in range(predict_days):
            # Делаем прогноз на следующий день
            next_pred = self.__model.predict(features.reshape(1, -1))[0]
            predictions.append(next_pred)
            
            # Обновляем признаки для следующего прогноза
            features = np.roll(features, 1)
            features[0] = next_pred

        return np.array(predictions)

    def __add_features(self, dataset:pd.DataFrame, lag_days:int):
        """Добавить признаки к данным"""
        # Создаем лаги 
        for i in range(1, lag_days+1):
            dataset[f'Close_{i}'] = dataset['Close'].shift(i)
        
        # Удаляем строки с NaN значениями
        dataset.dropna(inplace=True)

        return dataset

    def __holdout_split(self, dataset:pd.DataFrame, test_shape:int):
        """Разделение на обучающую и тестовую выборки"""
        train_df = dataset.iloc[:-test_shape]
        test_df  = dataset.iloc[-test_shape:]
        return train_df, test_df

    def __learn_model(self, dataset:pd.DataFrame):
        """Обучение модели Ridge"""
        # Разделение на признаки и таргет
        y = dataset['Close'].values
        
        x = dataset.copy()
        x.drop(['Close'], axis=1, inplace=True)
        x = x.values

        # Обучение
        self.__model.fit(x, y)

    def __correct_predict(self, last_price, predict):
        "Приведение прогноза к историческим данным"
        delta = abs(predict[0] - last_price)
        if predict[0] > last_price:
            predict -= delta
        else:
            predict += delta
        return predict