import os
from datetime import datetime

class Logger():

    def __init__(self):
        self.__path_to_file_log = './log/log.txt'
        self.__create_file()

    def to_log(self, msg_text:str):
        now = datetime.now()

        text = f"{now.strftime('%d.%m.%Y %H.%M.%S')} {msg_text}\n"

        with open(self.__path_to_file_log, 'a', encoding='utf-8') as file:
            file.write(text)

    def __create_file(self):

        dir_path = os.path.dirname(self.__path_to_file_log)

        # Создаём все недостающие директории
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        # Проверяем, существует ли файл
        if not os.path.isfile(self.__path_to_file_log):
            # Создаём пустой файл
            with open(self.__path_to_file_log, 'w'):
                pass
