import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import gradio as gr

from src.config import reset_folders
from src.chroma_service import add_to_chroma, delete_chroma_collection, prepare_prompt_from_query
from src.model_service import get_pipeline
from src.pdf_service import process_uploaded_pdf

os.environ["JPYPE_JVM_OPTIONS"] = "--enable-native-access=ALL-UNNAMED"

app = FastAPI()
qa_pipeline = get_pipeline()

PROMPT_TEMPLATE = """
You are a precise document QA assistant. Read the provided document context and answer the question.

Follow these rules step by step:
1. Ensure factual accuracy. Do not hallucinate.
2. Provide candidate answers only if they are supported by the context.
3. Strictly deduplicate answers: if multiple chunks contain the same answer, output it only once.
4. Return at most 3 unique candidate answers. Place the most suitable one at the top.
5. Answer in this format:
   "Answer: <answer>, File: <filename>, Page: <page-range>, Accuracy: <percentage>%"
6. Place a <END> at the end.

Question: For {manufacturer} {model_number}, what is the {query_attr}?

Document Context:
{context}

Answer:
"""

# ---------------- FastAPI Endpoints ----------------
def success_response(data: dict = None, message: str = ""):
    payload = {"status": "success", "message": message}
    if data:
        payload.update(data)
    return JSONResponse(content=payload, status_code=200)

def error_response(message: str, status_code: int = 500):
    return JSONResponse(content={"status": "error", "message": message}, status_code=status_code)

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        documents = process_uploaded_pdf(file)
        if not documents:
            return error_response("No documents attached.", status_code=400)

        add_to_chroma(documents)
        return success_response(message=f"Uploaded & Indexed {file.filename}")
    except Exception as e:
        return error_response(str(e), status_code=500)
    
@app.post("/ask_question")
async def ask_question(
    manufacturer: str = Form(""),
    model_number: str = Form(""),
    query_attr: str = Form(...),
    prompt_template: str = Form(PROMPT_TEMPLATE)
):
    try:
        prompt, hits = prepare_prompt_from_query(
            manufacturer, model_number, query_attr, prompt_template
        )
        if len(hits) == 0:
            return success_response(data={"answer": "No relevant information found in the documents.", "hits": hits})

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
        return success_response(message=f"Reset done: {reset_result}")
    except Exception as e:
        return error_response(str(e), status_code=500)

# ---------------- Gradio UI ----------------
def gr_upload(files):
    results = []
    if not files:
        return "⚠️ No files uploaded"

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

    return "\n".join(results)

def gr_reset():
    try:
        reset_result = reset_folders()
        delete_chroma_collection()
        return f"Reset done: {reset_result}", ""
    except Exception as e:
        return f"Reset failed: {e}", ""

def gr_ask(manufacturer, model_number, query_attr, prompt_template):
    try:
        prompt, hits = prepare_prompt_from_query(
            manufacturer, model_number, query_attr, prompt_template
        )
        if len(hits) == 0:
            return "No relevant information found in the documents.", ""
        else:
            generated_ids = qa_pipeline(prompt, max_new_tokens=256, do_sample=False)
            raw_output = generated_ids[0]["generated_text"].strip()
            answer = raw_output.split("<END>")[0].strip()
            return answer, hits
    except Exception as e:
        return f"Error: {str(e)}", ""

with gr.Blocks() as demo:
    gr.Markdown("## PDF QA Demo\nUpload PDFs → Query")
    prompt_state = gr.State(PROMPT_TEMPLATE)

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
        query_attr   = gr.Textbox(label="Query Attribute", placeholder="Enter attribute to query")
        ask_btn = gr.Button("Submit")
        answer = gr.Textbox(label="Answer", lines=8)
        hits_box = gr.Textbox(label="Context Hits (Full)", lines=20)
    
    with gr.Tab("Settings"):
        prompt_box = gr.Textbox(label="Prompt Template", value=PROMPT_TEMPLATE, lines=12)
        save_btn = gr.Button("Save Prompt")
    
    upload_btn.click(gr_upload, inputs=[pdfs], outputs=[upload_out])
    upload_btn.click(lambda: None, inputs=None, outputs=[pdfs])
    reset_btn.click(gr_reset, outputs=[reset_out, upload_out])
    
    ask_btn.click(
        gr_ask,
        inputs=[manufacturer, model_number, query_attr, prompt_box],
        outputs=[answer, hits_box]
    )

    save_btn.click(lambda new: new, inputs=[prompt_box], outputs=[prompt_state])

# Mount Gradio UI into FastAPI
app = gr.mount_gradio_app(app, demo, path="/ui", theme="soft")