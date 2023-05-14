from tqdm.auto import tqdm

import os
import pinecone
import yaml

from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import Pinecone
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


f = open('../params/credentials.yml', 'r')
credential = yaml.load(f, Loader=yaml.SafeLoader)
f.close()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', credential['OPENAI_API_KEY'])
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', credential['PINECONE_API_KEY'])
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', credential['PINECONE_API_ENV'])


f = open('../params/app.yml', 'r')
config = yaml.load(f, Loader=yaml.SafeLoader)
f.close()

loader = PyPDFLoader(config['PDF_SOURCE'][0])
data = loader.load()
print (f'There are {len(data)} document(s) in your data.')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
texts = text_splitter.split_documents(data)
print (f'Now you have {len(texts)} documents after chunked.')

embeddings = OpenAIEmbeddings(model='text-embedding-ada-002', openai_api_key=OPENAI_API_KEY)

try:
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_API_ENV
    )

    index_name = credential['PINECONE_INDEX']

    vectorstore = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)
    print('Vector upserted. Doing test question...')
    query = f"What is {config['PRODUCT']}?"
    print(query)
    
    vectors = vectorstore.similarity_search(query)
    
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    chain = load_qa_chain(llm, chain_type="stuff")
    
    qna_from_pdf = chain.run(input_documents=vectors, question=query)
    print(qna_from_pdf.strip())
    print('Done.')
except Exception as err:
    print(err)