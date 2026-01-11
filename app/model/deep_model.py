import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from .metrics import Metrics

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device):
        super(LSTMModel, self).__init__()
        self.__device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM слой
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, 
            batch_first=True, dropout=0.2
        )
        
        # Полносвязный слой для вывода
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Инициализация скрытых состояний
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(self.__device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(self.__device)
        
        # Проход через LSTM
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        
        # Берём последний выход LSTM
        out = self.fc(out[:, -1, :])
        return out
    
class Deep_model():

    def __init__(self):
        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.__scaler = MinMaxScaler(feature_range=(0, 1))
        
        self.__model = LSTMModel(
            input_size = 1,
            hidden_size = 30,
            num_layers = 3,
            output_size = 1,
            device = self.__device
        ).to(self.__device)

    def set_data(self, trading_history_df:pd.DataFrame):
        """Загрузка новых данных для прогнозирования"""
        self.__dataset_df = trading_history_df.copy()

    def test(self):
        """Выполнить обучение и тестирование модели"""
        forecast_days = 30
        test_df = self.__dataset_df.copy()

        # 1. стандартизация
        scale_data = self.__scale(test_df)
        
        # 2. Добавление признаков (временные лаги)
        feature_scale, target_scale = self.__add_features(
            dataset = scale_data,
            lag_days = forecast_days
        )

        # 3. Разделение на обучающую и тестовую выборки
        X_train, X_test = self.__holdout_split(
            dataset = feature_scale,
            test_shape = forecast_days
        )

        y_train, y_test = self.__holdout_split(
            dataset = target_scale,
            test_shape = forecast_days
        )

        # 3. Обучение модели
        self.__learn_model(
            X_train = X_train,
            y_train = y_train
        )

        # 4. Расчет прогноза
        features = X_test[0]
        predict = self.__predict(
            predict_days = forecast_days,
            features = features
        )

        # 5. Расчет метрик
        return Metrics(
            mae = mean_absolute_error(y_test, predict),
            rmse = mean_squared_error(y_test, predict),
            mape = mean_absolute_percentage_error(y_test, predict)
        )

    def fit_predict(self, predict_days=30):
        """Обучить модель и расчитать прогноз"""

        train_df = self.__dataset_df.copy()
        
        # 1. стандартизация
        scale_data = self.__scale(train_df)

        # 2. Добавление признаков (временные лаги)
        X_train, y_train = self.__add_features(
            dataset = scale_data,
            lag_days = predict_days
        )

        # 3. Обучение модели
        self.__learn_model(
            X_train = X_train,
            y_train = y_train
        )

        # 4. Расчет прогноза
        features = X_train[-1]
        predict = self.__predict(
            predict_days = predict_days,
            features = features
        )

        # 4. Приведение к историческим данным
        return self.__correct_predict(
            last_price = train_df.iloc[-1]['Close'], 
            predict = predict
        )

    def __predict(self, features, predict_days=30):
        """Расчитать прогноз"""
        self.__model.eval()
        predictions_scale = []
        
        # Текущая последовательность (будем её обновлять)
        current_seq = features.copy()
        
        with torch.no_grad():
            for _ in range(predict_days):
                # Преобразуем в тензор и переносим на устройство
                x = torch.tensor(current_seq, dtype=torch.float32).unsqueeze(0).to(self.__device)
                
                # Делаем прогноз на 1 шаг
                y_pred = self.__model(x)
                y_pred = y_pred.cpu().numpy().flatten()[0]
                
                # Сохраняем предсказание (ещё нормализованное)
                predictions_scale.append(y_pred)
    
                # Обновляем последовательность: сдвигаем окно, добавляем новое значение
                current_seq = np.append(current_seq[1:], [[y_pred]], axis=0)
        
    
        # Денормализуем предсказания
        predictions_scale = np.array(predictions_scale).reshape(-1, 1)
        predictions = self.__scaler.inverse_transform(predictions_scale)
        
        return predictions.flatten()

    def __scale(self, dataset:pd.DataFrame):
        """Выполнить стандартизацию данных"""
        prices_list = dataset['Close'].values.reshape(-1, 1)
        return self.__scaler.fit_transform(prices_list)

    def __add_features(self, dataset, lag_days:int):
        """Добавить признаки к данным"""
        # Создаем лаги 
        X, y = [], []
        lag_days += 1
        for i in range(len(dataset) - lag_days):
            X.append(dataset[i:i + lag_days])
            y.append(dataset[i + lag_days])
        return np.array(X), np.array(y)

    def __holdout_split(self, dataset, test_shape:int):
        """Разделение на обучающую и тестовую выборки"""
        train, test = dataset[:-test_shape], dataset[-test_shape:]
        return train, test

    def __learn_model(self, X_train, y_train):
        """Обучение модели LSTM"""
        # Переводим в тензоры
        X_train = self.__to_tensor(X_train)
        y_train = self.__to_tensor(y_train)
        
        # Функция потерь и оптимизатор
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.__model.parameters(), lr=0.001)
        
        # Количество эпох
        num_epochs = 1000
        
        for epoch in range(num_epochs):
            self.__model.train()
            optimizer.zero_grad()
            
            # Прямой проход
            outputs = self.__model(X_train)
            loss = criterion(outputs, y_train)
            
            # Обратный проход
            loss.backward()
            optimizer.step()

    def __to_tensor(self, array):
        return torch.tensor(array, dtype=torch.float32).to(self.__device)

    def __correct_predict(self, last_price, predict):
        "Приведение прогноза к историческим данным"
        delta = abs(predict[0] - last_price)
        if predict[0] > last_price:
            predict -= delta
        else:
            predict += delta
        return predict