from .bot_action import Actions

class Bot_chat():

    def __init__(self, chat_id, user_id):
        self.chat_id = chat_id
        self.user_id = user_id
        self.next_action = Actions.init

    def set_model(self, model):
        self.model = model