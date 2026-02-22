from celery import Celery

from src.chroma_service import add_to_specific, add_to_shared
from src.file_service import process_specific_saved, process_shared_saved

celery_app = Celery(
    "tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

@celery_app.task(bind=True)
def process_specific_task(self, file_path: str):
    try:
        documents = process_specific_saved(file_path)
        if not documents:
            raise ValueError("No documents extracted")
        add_to_specific(documents)
        return {"status": "done", "msg": f"Indexed {file_path}"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}

@celery_app.task(bind=True)
def process_shared_task(self, file_path: str):
    try:
        documents = process_shared_saved(file_path)
        if not documents:
            raise ValueError("No documents extracted")
        add_to_shared(documents)
        return {"status": "done", "msg": f"Indexed {file_path}"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}