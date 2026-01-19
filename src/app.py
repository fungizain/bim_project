import os
import re
from fastapi import FastAPI, UploadFile
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
2. Remove redundant or irrelevant information.
3. Provide at most 3 unique candidate answers. Place the most suitable one at the top.
4. If the question specifies a manufacturer and model number, try to match them explicitly.
5. Do not repeat identical answers from different chunks. Deduplicate before presenting.
6. After each answer, append the exact source location and accuracy percentage in this format:
   "Answer: <answer>, File: <filename>, Page: <page-range>, Accuracy: <percentage>%"

Question: {query}
Document Context:
{context}
Answer:
"""

# ---------------- Gradio UI ----------------
def gr_upload(files):
    results = []
    for f in files:
        with open(f.name, "rb") as fh:
            upload_file = UploadFile(filename=os.path.basename(f.name), file=fh)
            documents = process_uploaded_pdf(upload_file)

            if len(documents) > 0:
                add_to_chroma(documents)
                results.append(f"✅ Uploaded & Indexed {upload_file.filename}")
            else:
                results.append(f"⚠️ Failed {upload_file.filename}")
    return "\n".join(results)

def gr_reset():
    reset_result = reset_folders()
    delete_chroma_collection()
    return reset_result, ""

def gr_ask(query, prompt_template):
    try:
        prompt, hits = prepare_prompt_from_query(query, prompt_template)

        generated_ids = qa_pipeline(
            prompt,
            max_new_tokens=128,
            do_sample=False
        )
        answer = generated_ids[0]["generated_text"].strip()

        return answer, hits
    except Exception as e:
        return f"Error: {str(e)}", ""

with gr.Blocks() as demo:
    gr.Markdown("## 📄 PDF QA Demo\nUpload PDFs → Ask Questions → Reset")
    prompt_state = gr.State(PROMPT_TEMPLATE)

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
        hits_box = gr.Textbox(label="Context Hits (Full)", lines=20)
    
    with gr.Tab("Settings"):
        prompt_box = gr.Textbox(label="Prompt Template", value=PROMPT_TEMPLATE, lines=12)
        save_btn = gr.Button("💾 Save Prompt")

    upload_btn.click(gr_upload, inputs=[pdfs], outputs=[upload_out])
    upload_btn.click(lambda: None, inputs=None, outputs=[pdfs])
    reset_btn.click(gr_reset, outputs=[reset_out, upload_out])

    ask_btn.click(gr_ask, inputs=[query, prompt_box], outputs=[answer, hits_box])
    save_btn.click(lambda new: new, inputs=[prompt_box], outputs=[prompt_state])


# Mount Gradio UI into FastAPI
app = gr.mount_gradio_app(app, demo, path="/ui", theme="soft")