import telebot
import os

from telebot import apihelper
from model.bot_handler import Bot_handler

API_TOKEN =  os.environ.get('TELEGRAM_TOKEN')
bot = telebot.TeleBot(API_TOKEN, threaded=False)
bot.request_timeout = 100
apihelper.SESSION_TIMEOUT = 100

bot_handler = Bot_handler(bot)

@bot.message_handler(commands=['start'])
def send_welcome(message): 
    bot_handler.start_handler(message)

@bot.message_handler(func=lambda message: True)
def all(message):
    bot_handler.message_handler(message)

bot.polling(none_stop=True, interval=2)