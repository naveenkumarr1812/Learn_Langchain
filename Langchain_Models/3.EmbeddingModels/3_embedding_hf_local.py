
#For embedding a single query

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

text = "Delhi is the capital of India."

vector = embeddings.embed_query(text)

print(str(vector))





#For embedding multiple documents
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "Paris is the capital of France.",
    "London is the capital of England.",
    "Berlin is the capital of Germany.",
    "Delhi is the capital of India."
]           

vector = embeddings.embed_query(documents)

print(str(vector))