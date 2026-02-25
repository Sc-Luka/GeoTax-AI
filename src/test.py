from utils import *
from model import *

_, _, embeddings, _ = load_model_and_tokenizer()

retriever = buildRetriver(embeddings)

query = "რა არის პროცედურა საქონლის საბაჟო ღირებულების განსაზღვრისას?"

docs = retriever.invoke(f"query: {query}")
context = "\n\n".join([doc.page_content for doc in docs])

print(context)
