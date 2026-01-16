from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from src.config import CHROMA_PATH
from src.model_service import get_langchain_embedder

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

def get_chroma_collection() -> Chroma:
    return Chroma(
        collection_name="bim_project",
        embedding_function=get_langchain_embedder(),
        persist_directory=CHROMA_PATH
    )

def add_to_chroma(chunks: list[Document]):
    chroma_collection = get_chroma_collection()
    chroma_collection.add_documents(chunks)

def delete_chroma_collection():
    chroma_collection = get_chroma_collection()
    chroma_collection.delete_collection()
    print("🗑️ Deleted Chroma collection 'bim_project'")

def prepare_prompt_from_query(query: str, prompt_template: str, k: int = 3):
    chroma_collection = get_chroma_collection()
    hits = chroma_collection.similarity_search(query, k=k)

    context = "\n\n".join(
        [f"[chunk {h.metadata.get('chunk_id')}]\n{h.page_content}" for h in hits]
    )

    prompt = prompt_template.format(context=context, query=query)
    return prompt, hits