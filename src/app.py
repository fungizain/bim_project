import os
import uuid
import shutil
from celery.result import AsyncResult
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import gradio as gr

from src.config import (
    list_specific_folders,
    list_shared_folders,
    reset_specific_folders,
    reset_shared_folders
)
from src.chroma_service import (
    add_to_specific,
    add_to_shared,
    delete_specific,
    delete_shared,
    query_chroma
)
from src.model_service import model_predict
from src.file_service import process_specific_upload, process_shared_upload
from src.task import celery_app, process_specific_task, process_shared_task

os.environ["JPYPE_JVM_OPTIONS"] = "--enable-native-access=ALL-UNNAMED"

app = FastAPI()

# ---------------- FastAPI Endpoints ----------------
def success_response(msg: str = None, data: dict = None):
    detail = [{"type": "success"}]
    if msg:
        detail[0]["msg"] = msg
    if data:
        detail[0]["data"] = data
    return JSONResponse(content={"detail": detail}, status_code=200)

def error_response(msg: str, status_code: int = 400):
    detail = [{"msg": msg, "type": "error"}]
    return JSONResponse(content={"detail": detail}, status_code=status_code)

def file_to_tmp(file: UploadFile, job_id: str) -> str:
    file_path = f"/tmp/{job_id}_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return file_path

@app.post("/upload_specific_file")
async def upload_specific_file(file: UploadFile = File(...)):
    if not file:
        return error_response("No documents attached.", status_code=400)
    job_id = str(uuid.uuid4())
    file_path = file_to_tmp(file, job_id)
    process_specific_task.apply_async(args=[file_path], task_id=job_id)
    return success_response(data={"job_id": job_id, "status": "submitted"})

@app.post("/upload_shared_file")
async def upload_shared_file(file: UploadFile = File(...)):
    documents = process_shared_upload(file)
    if not documents:
        return error_response("No documents attached.", status_code=400)
    job_id = str(uuid.uuid4())
    file_path = file_to_tmp(file, job_id)    
    process_shared_task.apply_async(args=[file_path], task_id=job_id)
    return success_response(data={"job_id": job_id, "status": "submitted"})

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    result = AsyncResult(job_id, app=celery_app)
    data = {
        "job_id": job_id,
        "state": result.state,
        "result": result.result
    }
    return success_response(data=data)

@app.get("/list_specific")
async def list_specific():
    return success_response(data={"files": list_specific_folders()})

@app.get("/list_shared")
async def list_shared():
    return success_response(data={"files": list_shared_folders()})

@app.get("/reset_specific")
async def reset_specific():
    try:
        delete_specific()
        reset_result = reset_specific_folders()
        return success_response(msg=f"Reset done: {reset_result}")
    except Exception as e:
        return error_response(str(e), status_code=500)

@app.get("/reset_shared")
async def reset_shared():
    try:
        delete_shared()
        reset_result = reset_shared_folders()
        return success_response(msg=f"Reset done: {reset_result}")
    except Exception as e:
        return error_response(str(e), status_code=500)

@app.post("/ask_question")
async def ask_question(
    manufacturer: str = Form(""),
    model_number: str = Form(""),
    query_attr: str = Form(...),
):
    try:
        hits = query_chroma(manufacturer, model_number, query_attr)
        if len(hits) == 0:
            return success_response(data={"answer": "No relevant information found in the documents.", "hits": hits})
        
        answer = model_predict(manufacturer, model_number, query_attr, hits)
        return success_response(data={"answer": answer, "hits": hits})
    except Exception as e:
        return error_response(str(e), status_code=500)

# ---------------- Gradio UI ----------------
def gr_sp_upload(files) -> tuple[str, None]:
    results = []
    if not files:
        return "⚠️ No files uploaded", None

    for f in files:
        try:
            with open(f.name, "rb") as fh:
                upload_file = UploadFile(
                    filename=os.path.basename(f.name),
                    file=fh
                )
                documents = process_specific_upload(upload_file)

                if documents:
                    add_to_specific(documents)
                    results.append(f"Uploaded & Indexed {upload_file.filename}")
                else:
                    results.append(f"No documents extracted from {upload_file.filename}")
        except Exception as e:
            results.append(f"Error processing {f.name}: {e}")

    return "\n".join(results), None

def gr_sh_upload(files) -> tuple[str, None]:
    results = []
    if not files:
        return "⚠️ No files uploaded", None

    for f in files:
        try:
            with open(f.name, "rb") as fh:
                upload_file = UploadFile(
                    filename=os.path.basename(f.name),
                    file=fh
                )
                documents = process_shared_upload(upload_file)

                if documents:
                    add_to_shared(documents)
                    results.append(f"Uploaded & Indexed {upload_file.filename}")
                else:
                    results.append(f"No documents extracted from {upload_file.filename}")
        except Exception as e:
            results.append(f"Error processing {f.name}: {e}")

    return "\n".join(results), None

def gr_sp_reset() -> tuple[str, str]:
    try:
        reset_result = reset_specific_folders()
        delete_specific()
        return f"Reset done: {reset_result}", ""
    except Exception as e:
        return f"Reset failed: {e}", ""
    
def gr_sh_reset() -> tuple[str, str]:
    try:
        reset_result = reset_shared_folders()
        delete_shared()
        return f"Reset done: {reset_result}", ""
    except Exception as e:
        return f"Reset failed: {e}", ""

def gr_ask(manufacturer, model_number, query_attr) -> tuple[str, str]:
    try:
        hits = query_chroma(manufacturer, model_number, query_attr)
        if len(hits) == 0:
            return "No relevant information found in the documents.", ""

        answer = model_predict(manufacturer, model_number, query_attr, hits)
        return answer, hits
    except Exception as e:
        return f"Error: {str(e)}", ""

with gr.Blocks() as demo:
    gr.Markdown("## File QA Demo\nUpload files → Query")

    with gr.Tab("Upload Specific"):
        files = gr.File(label="Upload Specific Files", file_types=None, file_count="multiple")
        with gr.Row():
            with gr.Column():
                upload_sp_btn = gr.Button("Upload")
                upload_sp_out = gr.Textbox(label="Upload Result", lines=6)
            with gr.Column():
                reset_sp_btn = gr.Button("Reset")
                reset_sp_out = gr.Textbox(label="Reset Status", interactive=False)
    
    upload_sp_btn.click(gr_sp_upload, inputs=[files], outputs=[upload_sp_out, files])
    reset_sp_btn.click(gr_sp_reset, outputs=[reset_sp_out, upload_sp_out])

    with gr.Tab("Upload Shared"):
        files = gr.File(label="Upload Shared Files", file_types=None, file_count="multiple")
        with gr.Row():
            with gr.Column():
                upload_sh_btn = gr.Button("Upload")
                upload_sh_out = gr.Textbox(label="Upload Result", lines=6)
            with gr.Column():
                reset_sh_btn = gr.Button("Reset")
                reset_sh_out = gr.Textbox(label="Reset Status", interactive=False)
    
    upload_sh_btn.click(gr_sh_upload, inputs=[files], outputs=[upload_sh_out, files])
    reset_sh_btn.click(gr_sh_reset, outputs=[reset_sh_out, upload_sh_out])

    with gr.Tab("Query"):
        manufacturer = gr.Textbox(label="Manufacturer", placeholder="Enter manufacturer name")
        model_number = gr.Textbox(label="Model Number", placeholder="Enter model number")
        query_attr   = gr.Textbox(label="Query", placeholder="Enter attribute to query")
        ask_btn = gr.Button("Submit")
        answer = gr.Textbox(label="Answer", lines=8)
        hits_box = gr.Textbox(label="Context Hits (Full)", lines=20)
    
    ask_btn.click(
        gr_ask,
        inputs=[manufacturer, model_number, query_attr],
        outputs=[answer, hits_box]
    )

# Mount Gradio UI into FastAPI
app = gr.mount_gradio_app(app, demo, path="/ui", theme="soft")