import pytest
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

# ------------------------
# Mock dependencies
# ------------------------
@pytest.fixture(autouse=True)
def mock_dependencies(monkeypatch):
    monkeypatch.setattr("src.pdf_service.process_uploaded_pdf", lambda file: ["doc1", "doc2"])
    monkeypatch.setattr("src.chroma_service.add_to_chroma", lambda docs: None)
    # Mock prepare_prompt_from_query
    monkeypatch.setattr("src.chroma_service.prepare_prompt_from_query", lambda m, n, q, p: ("prompt", "hits"))
    # Mock qa_pipeline
    class DummyPipeline:
        def __call__(self, prompt, max_new_tokens, do_sample):
            return [{"generated_text": "Answer: 123, File: test.pdf, Page: 1-2, Accuracy: 100% <END>"}]
    monkeypatch.setattr("src.app.qa_pipeline", DummyPipeline())
    # Mock reset_folders
    monkeypatch.setattr("src.config.reset_folders", lambda: "folders reset")
    # Mock delete_chroma_collection
    monkeypatch.setattr("src.chroma_service.delete_chroma_collection", lambda: None)

# ------------------------
# Tests
# ------------------------

def test_upload_pdf_success(tmp_path):
    test_file = tmp_path / "test.pdf"
    test_file.write_text("dummy content")
    with open(test_file, "rb") as f:
        response = client.post("/upload_pdf", files={"file": ("test.pdf", f, "application/pdf")})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "Uploaded & Indexed" in data["message"]

def test_upload_pdf_no_docs(monkeypatch, tmp_path):
    monkeypatch.setattr("src.pdf_service.process_uploaded_pdf", lambda file: [])
    test_file = tmp_path / "empty.pdf"
    test_file.write_text("dummy content")
    with open(test_file, "rb") as f:
        response = client.post("/upload_pdf", files={"file": ("empty.pdf", f, "application/pdf")})
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error"
    assert "No documents attached" in data["message"]

def test_ask_question_success():
    response = client.post(
        "/ask_question",
        data={"manufacturer": "YORK", "model_number": "123", "query_attr": "Total Input Power"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["answer"]
    assert data["hits"] == "hits"

def test_reset_success():
    response = client.post("/reset")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "Reset done" in data["message"]