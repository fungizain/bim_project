from fastapi import FastAPI, UploadFile
import gradio as gr
import os

from src.folder_service import reset_folders
from src.faiss_service import process_and_update_index, prepare_prompt_from_query
from src.model_service import get_embedder, get_pipeline
from src.pdf_service import process_uploaded_pdf

os.environ["JPYPE_JVM_OPTIONS"] = "--enable-native-access=ALL-UNNAMED"

app = FastAPI()
embedder = get_embedder()
qa_pipeline = get_pipeline()

PROMPT_TEMPLATE = """
Read the following document and answer the question.
Keep the answer unique.

Follow this structured reasoning:
1. Identify key sections & main topics.
2. Extract essential points from each section.
3. Remove redundant information.
4. Ensure accuracy without hallucination.
5. Output only one concise answer.

Question:
{query}
Document:
{context}
"""

# ---------------- Gradio UI ----------------
def gr_upload(files):
    results = []
    for f in files:
        with open(f.name, "rb") as fh:
            upload_file = UploadFile(filename=os.path.basename(f.name), file=fh)

            # Step 1: OCR + 抽文字
            output_txt = process_uploaded_pdf(upload_file, lang="eng")

            # Step 2: 更新 FAISS index
            if output_txt:
                process_and_update_index(output_txt, embedder)
                results.append(f"✅ Uploaded & Indexed {upload_file.filename}")
            else:
                results.append(f"⚠️ Failed {upload_file.filename}")
    return "\n".join(results)

def gr_reset():
    return reset_folders(), ""

def gr_ask(query, prompt_template):
    try:
        # 用 faiss_service 封裝好嘅方法
        prompt, hits = prepare_prompt_from_query(query, embedder, prompt_template)
        answer = qa_pipeline(
            prompt,
            max_new_tokens=512,
            do_sample=False
        )[0]["generated_text"]

        # hits 全部顯示
        hits_text = "\n\n".join(
            [f"[{h.file_name} | chunk {h.chunk_id}]\n{h.text}" for h in hits]
        )

        return answer, hits_text
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