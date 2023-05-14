import os
import pinecone
import yaml


f = open('../params/credentials.yml', 'r')
credential = yaml.load(f, Loader=yaml.SafeLoader)
f.close()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', credential['OPENAI_API_KEY'])
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', credential['PINECONE_API_KEY'])
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', credential['PINECONE_API_ENV'])


try:
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
    indexs = pinecone.list_indexes()
    index = pinecone.Index(indexs[0])
    index.delete(deleteAll='true', namespace='')
    print('All vector data deleted.')
except Exception as err:
    print(err)