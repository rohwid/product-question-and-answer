from helper.chatbot import ChatBot
from helper.search import SearchAgent
from helper.db import user_manual_search

import yaml


f = open('params/app.yml', 'r')
config = yaml.load(f, Loader=yaml.SafeLoader)
f.close()


chat_session = True

print(f'Hello, Nice to meet you! I\'m bot that asigned for you.')
print(f"I will answer all your question about our product ({config['PRODUCT']}).")
print(f"If you wanna quit this chat, just type 'Bye!'.")
print(f'Let\'s begin the chat :)')

while chat_session:
    question = input("You  >>>  ")
    
    if question.lower() != 'bye!':
        pdf_result = user_manual_search(question)

        search = SearchAgent(question)
        serpapi_result = search.search_result()
        
        chatbot = ChatBot(question, pdf_result, serpapi_result)
        print(f'Bot  >>>  {chatbot.get_reply()}')
    
    if question.lower() == 'bye!':
        chat_session = False
    
print('Bot  >>>  Okay, see you!')