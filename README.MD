# Product QnA

This part explain about how this **Product QnA** works, how to setup and use this project. The code explanation were wrote sepately in `notebook/Product Question and Answer.ipynb`, it explain all about tools, function, module and etc for this project.

## Project Overview

<img src="img/langchain_architecture.png" width="1000"/>

Following the figure above, there are many implementations of the Langchain modules that described below from left to the right.

+ Firstly is about the PDF loader that loads the PDF file and then chunks it into many documents. Then continue by embedding each document content and upsert to the [Pinecone](https://www.pinecone.io/) database as vectors. The new PDF document can be added by put the PDF file location on a list inside `PDF_SOURCE` variable inside `params/app.yml` file. Then continue by open `query/upsert_pdf_vector_data.py` and change `pdf_list_number` variable with the number of the list where you put PDF file location in `params/app.yml`.
  
    **Other helpful scripts**:
    + `query/list_and_describe_index.py` can be use to check the list and the description of the Pinecone database index.
    + And `query/delete_all_vector_data.py` can be use to delete all vector data from the Pinecone database index.

+ Secondly is about performing Similarity Seach by Indexing the vector from Pinecone database and get the similar documents.
+ Third is the continuation of the Similarity Search that produces results about some similar documents. Then the QnA chain from Langchain module with OpenAI model will perform QnA by asking some question with **the human input** as the question.
+ Fourth is about agent selection that will select which agent that will perform. The Agent selection is depend on **the human input** or question, are there contains the **product name** that already write inside `params/app.yml` with `PRODUCT` variable. There are two agent, first agent corespond to do search with Google directly with **the human input** and the second agent corespond to to search with Google but add some keyword about the correlation with the product. The main purpose of this step is to reduce the Chatbot's reply time and SerpApi usage, if compared with the both Agent worked together.

    **Agent selection process**:
    + The selection process are depend to the `MUST_EXIST_PRODUCT_WORD` and the `ONE_OR_MORE_PRODUCT_WORD` variables in `params/app.yml`.
    + If the human input not contains word in `MUST_EXIST_PRODUCT_WORD`, then it will call the second agent.
    + If the human input contains word in `MUST_EXIST_PRODUCT_WORD` and not contain any words in `ONE_OR_MORE_PRODUCT_WORD` list, then it will call the second agent.
    + If the human input contains word in `MUST_EXIST_PRODUCT_WORD` and one of the words in `ONE_OR_MORE_PRODUCT_WORD` list, then it will call the first agent.
    + If the human input contains word in `MUST_EXIST_PRODUCT_WORD` and the `ONE_OR_MORE_PRODUCT_WORD` list variable is empty, then it will call the first agent. It means the product only use one word.

+ The result of the third and the fourth process will be summarized by prompting some command to instruct the OpenAI model.
+ The next step are prompting to instruct the OpenAI model to perform as Chatbot and consider the result of the summarization that comes from the information from the product User Manual's, Catalog's and etc also from the Agent that get some product information from the Google.
+ And the last step is the Chatbot will use Chain from Langchain library to relate or connect with the previous chat history and **the human input** that manage by Memory module from Langchain.