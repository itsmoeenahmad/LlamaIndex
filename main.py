# App Flow is like: Loading the pdfs, store it in DBs and then load the content from it.

# Importing Required Libraries
import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage 
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.postprocessor import SimilarityPostprocessor
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex

# Importing the environment variable - openai api key
load_dotenv()
open_api_key = os.getenv("OPENAI_API_KEY")
if open_api_key is not None:
    os.environ["OPENAI_API_KEY"] = open_api_key
#print('API Loaded')

# loading the data
documents = SimpleDirectoryReader("data").load_data()
#print('Loaded Data is ', documents)

# Indexing the data
index = VectorStoreIndex.from_documents(documents=documents, show_progress=True)
#print('Index is ', index)

# Searching the data
query_engine = index.as_query_engine()
#print('QueryEnding is: ',query_engine)

# Giving the query to query_engine
#response = query_engine.query('who is alice?')
# print('RESPONSE IS: ', response)

# Printing the final/best response and also seeing its source using:
#pprint_response(response=response, show_source=True)

# For Modifying the query_engine for checking the n numbers of documents using:
# In the below code we just set that: retrieve documents/source_data upto 3 documents.
retreiver = VectorIndexRetriever(
    index=index,
    similarity_top_k=3
)
query_engine = RetrieverQueryEngine(retriever=retreiver)

# Now asking a question and checking it.
# response = query_engine.query('who is alice?')
# pprint_response(response=response, show_source=True)

# For Getting those documents whose similarity score is greater than 80%(0.8) using:
post_processor = SimilarityPostprocessor(similarity_cutoff=0.70) # here similarity score is set to 70%
query_engine = RetrieverQueryEngine(
    retriever=retreiver,
    node_postprocessors=[post_processor]
)

# Now Checking
# response = query_engine.query('who is alice?')
# pprint_response(response=response, show_source=True)



# Till now our data(indexe) is stored in a memory which is good for smaller projects not for larger.
# So for larger projects we store our data in our system, its code is:

# Directory name within my system
PERSIST_DIR = './storage'
# Checking is the directory exist or not, if exist then loading the data and if not then storing the data.
if not os.path.exists(PERSIST_DIR):
    print('not exist')
    print('Lets add it.')
    # loading the documents
    documents = SimpleDirectoryReader('data').load_data()
    # indexing the documents
    index = VectorStoreIndex.from_documents(documents=documents)

    # Now, Storing it inside the directory - in my system
    index.storage_context.persist(persist_dir=PERSIST_DIR)

else:
    print('exist')
    print('Lets load it')
    # loading the existing data
    storage_content = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context=storage_content)



# Working with llama cloud:
# First, create an index with name on llama-cloud then, calling the testing-index from the llama-cloud:
index = LlamaCloudIndex(
  name="testing-index",
  project_name="Default",
  project_id='606a8199-e55e-4614-8694-b8549277b562',
  organization_id="6a842057-c20a-49e1-8ee3-d5b12bdebf33",
  api_key="llx-xPDw47BTNY31TT5H5Kdn9djJa5P8AEiELyBcWTwZd8CxqB7f", # API_KEY of llama_cloud
)

query='who is moeen ahmad?'
response = index.as_query_engine().query(query)
print('response is: ',response)