import os
import yaml

from langchain.memory import ConversationBufferMemory
from langchain import LLMChain
from langchain.llms import OpenAI
from langchain import PromptTemplate


f = open('params/credentials.yml', 'r')
credential = yaml.load(f, Loader=yaml.SafeLoader)
f.close()

f = open('params/app.yml', 'r')
config = yaml.load(f, Loader=yaml.SafeLoader)
f.close()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', credential['OPENAI_API_KEY'])

class ChatBot:
    def __init__(self, question, pdf_result, serpapi_result):
        self._question = question
        self._pdf_result = pdf_result
        self._serpapi_result = serpapi_result
        
    def get_reply(self):
        sumarization_result = self.summarize()
        chat_reply = self.set_reply(sumarization_result)
        
        return chat_reply
    
    def summarize(self):
        template = """
        %INSTRUCTIONS:
        Please summarize the following piece of text.

        %TEXT:
        {text}
        """

        # Create a LangChain prompt template that we can insert values to later
        prompt = PromptTemplate(
            input_variables=["text"],
            template=template,
        )

        confusing_text = """{agent_search}
        
        {pdf_result}
        """.format(
            agent_search = self._serpapi_result.strip(),
            pdf_result = self._pdf_result.strip()
        )
        
        summarize_prompt = prompt.format(text=confusing_text)
        llm = OpenAI(temperature=0, model_name='text-davinci-003', openai_api_key=OPENAI_API_KEY)
        summarize_result = llm(summarize_prompt)
        
        return summarize_result
    
    def set_reply(self, summarize_result):
        command = """
        %INSTRUCTIONS:
        You are very helpful chatbot.
        Your goal are help users to know more about the {product}.
        This following piece of text are some information that you need to consider too.

        %TEXT: {text}
        """

        prompt = PromptTemplate(
            input_variables=["product", "text"],
            template=command,
        )

        command_prompt = prompt.format(product=config['PRODUCT'], text=summarize_result)

        chat = """
        {chat_history}
        Human: {human_input}
        Chatbot:"""

        template = command_prompt + chat

        prompt = PromptTemplate(
            input_variables=["chat_history", "human_input"], 
            template=template
        )

        memory = ConversationBufferMemory(memory_key="chat_history")

        chat_chain = LLMChain(
            llm=OpenAI(model_name='text-davinci-003', openai_api_key=credential['OPENAI_API_KEY']), 
            prompt=prompt, 
            verbose=False, 
            memory=memory
        )
        
        chat_reply = chat_chain.predict(human_input=f'{self._question}')
        
        return chat_reply.strip()
    