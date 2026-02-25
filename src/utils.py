import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

folder_path = "data"


def documentsLoader(folder_path):
    documents = []

    if not os.path.exists(folder_path):
        print(f" {folder_path} ფაილი ვერ მოიძებნა!")
        return []

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            loader = PyPDFLoader(file_path)
            pages = loader.load()

            for page in pages:
                page.metadata["source"] = filename
                documents.append(page)

    return documents


def buildVectorDB(embeddings):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200
    )

    split_doc = text_splitter.split_documents(documentsLoader(folder_path))

    vector_db = Chroma.from_documents(
        documents=split_doc,
        embedding=embeddings,
        persist_directory="./vector_db/"
    )

    return vector_db


def buildRetriver(embeddings):
    if os.path.exists("./vector_db/"):
        print("--- vector_db ნაპოვნია, ვტვირთავ... ---")
        vector_db = Chroma(
            persist_directory="./vector_db/",
            embedding_function=embeddings
        )
    else:
        print("--- ბაზა არ არსებობს, ვიწყებ შექმნას... ---")
        vector_db = buildVectorDB(embeddings)

    return vector_db.as_retriever(search_kwargs={"k": 3})
