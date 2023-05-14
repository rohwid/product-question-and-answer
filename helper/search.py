import os
import re
import yaml

from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from langchain.agents import Tool
from langchain import SerpAPIWrapper


f = open('params/credentials.yml', 'r')
credential = yaml.load(f, Loader=yaml.SafeLoader)
f.close()

f = open('params/app.yml', 'r')
config = yaml.load(f, Loader=yaml.SafeLoader)
f.close()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', credential['OPENAI_API_KEY'])

os.environ['SERPAPI_API_KEY'] = credential['SERPAPI_API_KEY']
os.environ['REQUESTS_CA_BUNDLE'] = credential['REQUESTS_CA_BUNDLE']


class SearchAgent:
    def __init__(self, question):
        self._question = question
        
    def search_result(self):
        must_exists_words = config['MUST_EXIST_PRODUCT_WORD']
        optional_words = config['ONE_OR_MORE_PRODUCT_WORD']

        clean_question = re.sub(r'[^\w\s]', '', self._question.lower())
        question_words = clean_question.strip().split()

        match_set = set()

        must_exist = 0
        opt_exist = 0

        for word in question_words:
            if word not in match_set:
                for exist in must_exists_words:
                    if word == exist.lower():
                        match_set.add(exist)
                        must_exist += 1
            
            if word not in match_set and optional_words[0]:
                for opt in optional_words:
                    if word == opt.lower():
                        match_set.add(opt)
                        opt_exist += 1

        if len(must_exists_words) == must_exist and optional_words[0] and opt_exist >= 1:
            result = self.serpapi_search_all()
            return result

        if len(must_exists_words) == must_exist and optional_words[0] and opt_exist == 0:
            result = self.serpapi_search_related()
            return result

        if len(must_exists_words) == must_exist and not optional_words[0]:
            result = self.serpapi_search_all()
            return result

        if len(must_exists_words) != must_exist:
            result = self.serpapi_search_related()
            return result

    def serpapi_search_all(self):
        llm = OpenAI(temperature=0, model_name='text-davinci-003', openai_api_key=OPENAI_API_KEY)
        
        search = SerpAPIWrapper()
        
        search_toolkit = [
            Tool(
                name = 'Search',
                func=search.run,
                description='useful for when you need to search google to answer questions about current events'
            )
        ]
        
        agent = initialize_agent(
            search_toolkit, 
            llm, 
            agent='zero-shot-react-description', 
            verbose=False, 
            return_intermediate_steps=True
        )
        
        response = agent({'input': f'{self._question}'})
        
        return response['output']
        
    def serpapi_search_related(self):
        llm = OpenAI(temperature=0, model_name='text-davinci-003', openai_api_key=OPENAI_API_KEY)
        
        search = SerpAPIWrapper()
        
        search_toolkit = [
            Tool(
                name = 'Search',
                func=search.run,
                description='useful for when you need to search google to answer questions about current events'
            )
        ]
        
        agent = initialize_agent(
            search_toolkit, 
            llm, 
            agent='zero-shot-react-description', 
            verbose=False, 
            return_intermediate_steps=True
        )
        
        response = agent(
            {
                'input': f"{self._question} and what is the correlation with {config['PRODUCT']}"
            }
        )
        
        return response['output']