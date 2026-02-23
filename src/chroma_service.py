import chromadb
from langchain_core.documents import Document

from src.config import CHROMA_PATH
from src.model_service import get_embedder

client = chromadb.PersistentClient(path=CHROMA_PATH)

def get_collection(name: str):
    return client.get_or_create_collection(
        name=name,
        embedding_function=get_embedder()
    )

def add_to_collection(chunks: list[Document], name: str):
    collection = get_collection(name)
    collection.upsert(
        documents=[chunk.page_content for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
        ids=[chunk.metadata.get("chunk_id") for chunk in chunks]
    )

def delete_collection(name: str):
    client.delete_collection(name=name)

def get_specific():
    return get_collection("specific")

def get_shared():
    return get_collection("shared")

def add_to_specific(chunks: list[Document]):
    add_to_collection(chunks, "specific")

def add_to_shared(chunks: list[Document]):
    add_to_collection(chunks, "shared")

def delete_specific():
    delete_collection("specific")

def delete_shared():
    delete_collection("shared")

def query_collection(
        collection: chromadb.Collection,
        query_text: str,
        filters: dict,
        k: int = 5
    ) -> str:
    results = collection.query(
        query_texts=[query_text],
        n_results=k,
        where_document=filters if filters else None,
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    # distances = results["distances"][0]
    hits = "\n\n".join(
        [
            f"Ref: {m.get('source')} | pages: {m.get('pages')}\n"
            f"{doc}"
            for m, doc in zip(metadatas, documents)
        ]
    )

    return hits

def query_chroma(manufacturer: str, model_number: str, query_attr: str, k: int = 5) -> str:    
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

    specific_hits = query_collection(get_specific(), query_text, filters, k)
    shared_hits = query_collection(get_shared(), query_text, filters, k)

    if specific_hits and shared_hits:
        return (
            "\n\n=== PRIORITY: SPECIFIC COLLECTION ===\n\n" + specific_hits +
            "\n\n=== FALLBACK: SHARED COLLECTION ===\n\n" + shared_hits
        )
    elif specific_hits:
        return specific_hits
    elif shared_hits:
        return shared_hits
    else:
        return "Not Found"