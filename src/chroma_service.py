import chromadb
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHROMA_PATH
from src.model_service import get_embedder

client = chromadb.PersistentClient(path=CHROMA_PATH)

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
        embedding_function=get_embedder()
    )
    return collection

def add_to_chroma(chunks: list[Document]):
    collection = get_chroma_collection()
    collection.upsert(
        documents=[chunk.page_content for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
        ids=[chunk.metadata.get("chunk_id") for chunk in chunks]
    )

def delete_chroma_collection():
    client.delete_collection(name="bim_project")

def query_chroma(manufacturer: str, model_number: str, query_attr: str, k: int = 5) -> str:
    collection = get_chroma_collection()
    
    query_parts = [manufacturer.strip(), model_number.strip(), query_attr.strip()]
    query_text = " ".join([q for q in query_parts if q])
    filters = {}
    if manufacturer.strip():
        filters["$regex"] = f"(?i){manufacturer.strip()}"
    if model_number.strip():
        if "$regex" in filters:
            filters["$and"] = [
                {"$regex": f"(?i){manufacturer.strip()}"},
                {"$regex": f"(?i){model_number.strip()}"}
            ]
            filters.pop("$regex")
        else:
            filters["$regex"] = f"(?i){model_number.strip()}"
    print(f"Querying ChromaDB with text: {query_text} and filters: {filters}")

    results = collection.query(
        query_texts=[query_text],
        n_results=k,
        where_document=filters if filters else None,
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    hits = "\n\n".join(
        [
            f"[chunk {m.get('chunk_id')} | distance: {dist:.4f}]\n"
            f"source: {m.get('source')} | pages: {m.get('page_start')}-{m.get('page_end')}\n"
            f"{doc}"
            for m, doc, dist in zip(metadatas, documents, distances)
        ]
    )

    return hits