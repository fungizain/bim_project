import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import gradio as gr

from src.config import reset_folders
from src.chroma_service import add_to_chroma, delete_chroma_collection, query_chroma
from src.model_service import get_pipeline
from src.pdf_service import process_uploaded_pdf
from src.prompt_service import (
    create_session,
    change_system_template,
    get_system_prompt,
    prepare_prompt,
    prepare_prompt_with_template
)

os.environ["JPYPE_JVM_OPTIONS"] = "--enable-native-access=ALL-UNNAMED"

app = FastAPI()
qa_pipeline = get_pipeline()

# ---------------- FastAPI Endpoints ----------------
def success_response(msg: str, data: dict = None):
    detail = [{"msg": msg, "type": "success"}]
    if data:
        detail[0]["data"] = data
    return JSONResponse(content={"detail": detail}, status_code=200)

def error_response(message: str, status_code: int = 400):
    detail = [{"msg": message, "type": "error"}]
    return JSONResponse(content={"detail": detail}, status_code=status_code)

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        documents = process_uploaded_pdf(file)
        if not documents:
            return error_response("No documents attached.", status_code=400)

        add_to_chroma(documents)
        return success_response(msg=f"Uploaded & Indexed {file.filename}")
    except Exception as e:
        return error_response(str(e), status_code=500)
    
@app.post("/ask_question")
async def ask_question(
    manufacturer: str = Form(""),
    model_number: str = Form(""),
    query_attr: str = Form(...),
    prompt_template: str = Form(None)
):
    try:
        hits = query_chroma(manufacturer, model_number, query_attr)
        if len(hits) == 0:
            return success_response(data={"answer": "No relevant information found in the documents.", "hits": hits})
        
        prompt = prepare_prompt_with_template(
            manufacturer, model_number, query_attr, hits, prompt_template
        )
        generated_ids = qa_pipeline(prompt, max_new_tokens=256, do_sample=False)
        raw_output = generated_ids[0]["generated_text"].strip()
        answer = raw_output.split("<END>")[0].strip()

        return success_response(data={"answer": answer, "hits": hits})
    except Exception as e:
        return error_response(str(e), status_code=500)
    
@app.post("/reset")
async def reset():
    try:
        reset_result = reset_folders()
        delete_chroma_collection()
        return success_response(msg=f"Reset done: {reset_result}")
    except Exception as e:
        return error_response(str(e), status_code=500)

# ---------------- Gradio UI ----------------
def gr_upload(files) -> tuple[str, None]:
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
                documents = process_uploaded_pdf(upload_file)

                if documents:
                    add_to_chroma(documents)
                    results.append(f"Uploaded & Indexed {upload_file.filename}")
                else:
                    results.append(f"No documents extracted from {upload_file.filename}")
        except Exception as e:
            results.append(f"Error processing {f.name}: {e}")

    return "\n".join(results), None

def gr_reset() -> tuple[str, str]:
    try:
        reset_result = reset_folders()
        delete_chroma_collection()
        return f"Reset done: {reset_result}", ""
    except Exception as e:
        return f"Reset failed: {e}", ""

def gr_ask(session_id, manufacturer, model_number, query_attr) -> tuple[str, str]:
    try:
        hits = query_chroma(manufacturer, model_number, query_attr)
        if len(hits) == 0:
            return "No relevant information found in the documents.", ""
        else:
            prompt = prepare_prompt(session_id, manufacturer, model_number, query_attr, hits)
            generated_ids = qa_pipeline(prompt, max_new_tokens=256, do_sample=False)
            raw_output = generated_ids[0]["generated_text"].strip()
            answer = raw_output.split("<END>")[0].strip()
            return answer, hits
    except Exception as e:
        return f"Error: {str(e)}", ""

def gr_update_prompt(session_id, new_prompt) -> str:
    change_system_template(session_id, new_prompt)
    return new_prompt

with gr.Blocks() as demo:
    gr.Markdown("## PDF QA Demo\nUpload PDFs → Query")
    prompt_state = gr.State(create_session())

    with gr.Tab("Upload"):
        pdfs = gr.File(label="Upload PDFs", file_types=[".pdf"], file_count="multiple")
        with gr.Row():
            with gr.Column():
                upload_btn = gr.Button("Upload")
                upload_out = gr.Textbox(label="Upload Result", lines=6)
            with gr.Column():
                reset_btn = gr.Button("Reset")
                reset_out = gr.Textbox(label="Reset Status", interactive=False)
    
    with gr.Tab("Ask"):
        manufacturer = gr.Textbox(label="Manufacturer", placeholder="Enter manufacturer name")
        model_number = gr.Textbox(label="Model Number", placeholder="Enter model number")
        query_attr   = gr.Textbox(label="Query", placeholder="Enter attribute to query")
        ask_btn = gr.Button("Submit")
        answer = gr.Textbox(label="Answer", lines=8)
        hits_box = gr.Textbox(label="Context Hits (Full)", lines=20)
    
    with gr.Tab("Settings"):
        prompt_box = gr.Textbox(
            label="Prompt Template", value=get_system_prompt(prompt_state), lines=12
        )
        save_btn = gr.Button("Save Prompt")
    
    upload_btn.click(gr_upload, inputs=[pdfs], outputs=[upload_out, pdfs])
    reset_btn.click(gr_reset, outputs=[reset_out, upload_out])
    ask_btn.click(
        gr_ask,
        inputs=[prompt_state, manufacturer, model_number, query_attr],
        outputs=[answer, hits_box]
    )
    save_btn.click(
        gr_update_prompt,
        inputs=[prompt_state, prompt_box],
        outputs=[prompt_box]
    )

# Mount Gradio UI into FastAPI
app = gr.mount_gradio_app(app, demo, path="/ui", theme="soft")