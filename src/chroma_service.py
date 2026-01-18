import chromadb
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHROMA_PATH
from src.model_service import get_langchain_embedder

client = chromadb.Client()

def get_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False
    )

def split_documents(documents: list[Document]) -> list[Document]:
    text_splitter = get_text_splitter()
    return text_splitter.split_documents(documents)

def get_chroma_collection():
    collection = client.get_or_create_collection(
        name="bim_project",
        embedding_function=get_langchain_embedder(),
        persist_directory=CHROMA_PATH
    )
    return collection

def add_to_chroma(chunks: list[Document]):
    collection = get_chroma_collection()
    collection.upsert(
        documents=[chunk.page_content for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
        ids=[chunk.metadata.get("chunk_id") for chunk in chunks]
    )
    collection.persist()

def delete_chroma_collection():
    client.delete_collection(name="bim_project")

def prepare_prompt_from_query(query: str, prompt_template: str, k: int = 3):
    collection = get_chroma_collection()
    hits = collection.query(
        query_texts=[query],
        n_results=k,
        # where={"metadata_field": "is_equal_to_this"}, # optional filter
        # where_document={"$contains":"search_string"}  # optional filter
    )

    context = "\n\n".join(
        [f"[chunk {h.metadata.get('chunk_id')}]\n{h.page_content}" for h in hits]
    )

    prompt = prompt_template.format(context=context, query=query)
    return prompt, hits