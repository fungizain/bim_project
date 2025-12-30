from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import shutil
from pathlib import Path

from src.pdf_service import process_uploaded_pdf
from src.faiss_service import process_and_update_index, prepare_prompt_from_query
from src.model_service import get_embedder, get_pipeline

import gradio as gr
import os

app = FastAPI()

embedder = get_embedder()
qa_pipeline = get_pipeline()

# ---------------- FastAPI Endpoints ----------------
@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Step 1: OCR + 抽文字
        full_text, output_pdf, output_txt= process_uploaded_pdf(file, lang="eng")
        # Step 2: 更新 FAISS index (直接用 output_txt)
        result = process_and_update_index(output_txt, embedder)

        return {
            "status": result["status"],
            "file": file.filename,
            "ocr_pdf": str(output_pdf) if output_pdf else None,
            "text_file": str(result["text_file"]) if result["text_file"] else None,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/ask/")
async def ask_question(query: str = Form(...)):
    try:
        # 用 faiss_service 封裝好嘅方法
        prompt, hits = prepare_prompt_from_query(query, embedder)
        answer = qa_pipeline(
            prompt,
            max_new_tokens=256,
            do_sample=False
        )[0]["generated_text"]

        return {
            "query": query,
            "answer": answer,
            "context_hits": [
                {"file": h.file_name, "chunk": h.chunk_id, "text": h.text[:200]}
                for h in hits
            ],
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ---------------- Gradio UI ----------------
def gr_upload(files):
    results = []
    for f in files:
        # 將 Gradio File 轉成 FastAPI UploadFile
        with open(f.name, "rb") as fh:
            upload_file = UploadFile(filename=os.path.basename(f.name), file=fh)

            # Step 1: OCR + 抽文字
            full_text, output_pdf, output_txt = process_uploaded_pdf(upload_file, lang="eng")

            # Step 2: 更新 FAISS index
            if output_txt:
                process_and_update_index(output_txt, embedder)
                results.append(f"✅ Uploaded & Indexed {upload_file.filename}")
            else:
                results.append(f"⚠️ Failed {upload_file.filename}")
    return "\n".join(results)


def gr_ask(query):
    try:
        # 用 faiss_service 封裝好嘅方法
        prompt, hits = prepare_prompt_from_query(query, embedder)
        answer = qa_pipeline(
            prompt,
            max_new_tokens=128,
            do_sample=False
        )[0]["generated_text"]

        return answer
    except Exception as e:
        return f"Error: {str(e)}"

def gr_reset():
    for folder in ["src/upload_pdfs", "src/output_pdfs", "src/output_texts", "src/faiss_store"]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            os.makedirs(folder, exist_ok=True)
    return "✅ Reset success", ""

with gr.Blocks() as demo:
    gr.Markdown("## 📄 PDF QA Demo\nUpload PDFs → Ask Questions → Reset")

    with gr.Tab("Upload"):
        pdfs = gr.File(label="Upload PDFs", file_types=[".pdf"], file_count="multiple")
        with gr.Row():
            with gr.Column():
                upload_btn = gr.Button("📤 Upload")
                upload_out = gr.Textbox(label="Upload Result", lines=6)
            with gr.Column():
                reset_btn = gr.Button("🗑️ Reset")
                reset_out = gr.Textbox(label="Reset Status", interactive=False)

    with gr.Tab("Ask"):
        query = gr.Textbox(label="❓ Ask a question", placeholder="Type your question here...")
        ask_btn = gr.Button("🔍 Submit")
        answer = gr.Textbox(label="Answer", lines=8)

    upload_btn.click(gr_upload, inputs=[pdfs], outputs=[upload_out])
    upload_btn.click(lambda: None, inputs=None, outputs=[pdfs])
    ask_btn.click(gr_ask, inputs=[query], outputs=[answer])
    reset_btn.click(gr_reset, outputs=[reset_out, upload_out])

# Mount Gradio UI into FastAPI
app = gr.mount_gradio_app(app, demo, path="/ui", theme="soft")