class Metrics():
    
    def __init__(self, mae=0.0, rmse=0.0, mape=0.0):
        self.mae = mae
        self.rmse = rmse
        self.mape = mape

    def __str__(self):
        return f'MAE  = {self.mae}\nRMSE = {self.rmse}\nMAPE = {self.mape}'