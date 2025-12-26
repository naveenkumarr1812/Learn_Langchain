from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=32)

documents = [
    "Paris is the capital of France.",
    "London is the capital of England.",
    "Berlin is the capital of Germany.",
    "Delhi is the capital of India."
]

result = embeddings.embed_documents()

print(str(result))