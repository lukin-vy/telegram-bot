from .bot_action import Actions
from .bot_model import Bot_model
from .bot_chat import Bot_chat
from . import LOGGER

class Bot_handler():

    def __init__(self, api_bot):
        self.__api_bot = api_bot
        # "Временная БД"
        self.__bot_chats = dict()

    def start_handler(self, message):
        # Создаем объект чата с пользователем
        chat_id = message.chat.id
        user_id = message.from_user.id
        
        current_chat = Bot_chat(
            chat_id=chat_id,
            user_id=user_id
        )
        self.__bot_chats[chat_id] = current_chat
        
        response_text = 'Привет! Я ваш новый Telegram-бот для прогнозирования цен акций на Американской фондовой бирже.'
        self.__api_bot.reply_to(message, response_text)

        response_text = "Укажите тикер компании:"
        self.__api_bot.send_message(chat_id, response_text)
        current_chat.next_action = Actions.get_tiker

    def message_handler(self, message):
        # Получаем объект чата
        chat_id = message.chat.id
        user_id = message.from_user.id
        
        chat = self.__bot_chats.get(chat_id)
        if not chat:
            response_text = "Введите команду /start"
            self.__api_bot.send_message(chat_id, response_text)
            return

        if message.from_user.is_bot:
            return

        if chat.next_action == Actions.get_tiker:
            LOGGER.to_log(f'ID пользователя {user_id}')
            
            self.__api_bot.send_chat_action(chat_id, 'typing')
            # Обрабатываем получение тикера
            try:
                model = Bot_model(ticker = message.text)
                LOGGER.to_log(f'Тикер компании {message.text}')
                chat.set_model(model)
                cahart_img = model.predict_price()
            except ValueError:
                response_text = "Некорректно указан тикер компании.\nПовторите попытку:"
                self.__api_bot.send_message(chat_id, response_text)
                return
            
            self.__api_bot.send_photo(
                chat_id=chat_id,
                photo=cahart_img,
                caption=f'Прогноз по цене акции {message.text}'
            )
            
            response_text = "Укажите сумму инвестирования:"
            self.__api_bot.send_message(chat_id, response_text)
            chat.next_action = Actions.get_deals

            return

        if chat.next_action == Actions.get_deals:
            self.__api_bot.send_chat_action(chat_id, 'typing')

            # Обрабатываем получение инвест суммы
            try:
                invest = float(message.text)
            except ValueError:
                response_text = "Некорректно введено число.\nПовторите попытку:"
                self.__api_bot.send_message(chat_id, response_text)
                return
                
            profit_text, profit_chart = chat.model.get_profit(invest)

            self.__api_bot.send_photo(
                chat_id=chat_id,
                photo=profit_chart,
                caption=profit_text
            )
            
            response_text = "*Не является индивидуальной инвестиционной рекомендацией\nПроект разработан в рамках сессионого задания."
            self.__api_bot.send_message(chat_id, response_text)
            chat.next_action = Actions.init

            return

        if chat.next_action == Actions.init:
            response_text = "Укажите тикер компании:"
            self.__api_bot.send_message(chat_id, response_text)
            chat.next_action = Actions.get_tiker

            return