import os
from chromadb.utils import embedding_functions
from openai_harmony import (
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    Role,
    SystemContent,
    load_harmony_encoding,
    ReasoningEffort
)
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GptOssForCausalLM,
    pipeline
)

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"

env = os.getenv("APP_ENV")
if env == "prod":
    print("Running in production mode.")
    # LLM_MODEL = "nvidia/Nemotron-Orchestrator-8B"
    LLM_MODEL = "openai/gpt-oss-20b"
else:
    print("Running in development mode.")
    LLM_MODEL = "Qwen/Qwen3-0.6B"

enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
autoModel = GptOssForCausalLM if "gpt-oss" in LLM_MODEL else AutoModelForCausalLM
model = autoModel.from_pretrained(
    LLM_MODEL,
    device_map="auto",
    dtype="auto",
    trust_remote_code=True
)

qa_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False
)

def get_tokenizer():
    return tokenizer

def get_embedder():
    return embedder

def model_predict_from_prompt(prompt: str):
    generated_ids = qa_pipeline(prompt, max_new_tokens=1024, do_sample=False)
    raw_output = generated_ids[0]["generated_text"].strip()
    answer = raw_output.split("<END>")[0].strip()
    return answer

def prepare_convo(
        manufacturer: str,
        model_number: str,
        query_attr: str,
        hits: str
) -> Conversation:
    system_message = (
    SystemContent.new()
        .with_model_identity("You are a retrieval-augmented assistant.")
        .with_reasoning_effort(ReasoningEffort.LOW)
        .with_conversation_start_date("2026-01-26")
        .with_knowledge_cutoff("2024-06")
        .with_required_channels(["final"])
    )

    # task_ins = (
    #     "Task:\n"
    #     "- Read the document content (up to 10 chunks provided).\n"
    #     f"- Extract the value of the attribute {query_attr} from each relevant chunk.\n"
    #     "- Always prioritize chunks from the 'SPECIFIC COLLECTION'.\n"
    #     "- If valid answers are found in 'SPECIFIC COLLECTION', return them.\n"
    #     "- If no valid answers are found in 'SPECIFIC COLLECTION', then check 'SHARED COLLECTION'.\n"
    #     "- If both collections contain valid answers, return results from both, clearly separated.\n"
    #     "- If multiple possible answers exist, return all unique values found, up to 5 in total.\n"
    # )

    # constraints_ins = "Constraints:\n"
    # if manufacturer and model_number:
    #     constraints_ins += f"- Ensure the answer matches manufacturer {manufacturer} and model number {model_number}.\n"
    # elif manufacturer:
    #     constraints_ins += f"- Ensure the answer matches manufacturer {manufacturer}.\n"
    # elif model_number:
    #     constraints_ins += f"- Ensure the answer matches model number {model_number}.\n"
    # constraints_ins += (
    #     "- Check each chunk individually and extract candidate values.\n"
    #     "- Do not skip chunks even if one answer seems highly confident.\n"
    #     "- Absolutely do not return duplicate answers.\n"
    #     "- Sort answers by confidence level from highest to lowest.\n"
    # )

    # output_ins = (
    #     "Output Format:\n"
    #     "- Each answer must be formatted strictly as:\n"
    #     "<value> (<confidence>%) [Ref: <filename> page <page> line <line>]\n"
    #     "- Answers found in 'SPECIFIC COLLECTION' is roughly 15% more reliable than 'SHARED COLLECTION'.\n"
    #     "- Confidence maximum is 100%.\n"
    #     "- If no answer is found, answer: Not Found"
    # )

    task_ins = (
        "Task:\n"
        f"- Extract the attribute {query_attr} from up to 10 chunks.\n"
        "- Prioritize 'SPECIFIC COLLECTION'.\n"
        "- If no valid answer in SPECIFIC, check 'SHARED COLLECTION'.\n"
        "- If both collections have answers, return both (clearly separated).\n"
        "- Return up to 5 unique values.\n"
    )

    constraints_ins = "Constraints:\n"
    if manufacturer and model_number:
        constraints_ins += f"- Match manufacturer {manufacturer} and model {model_number}.\n"
    elif manufacturer:
        constraints_ins += f"- Match manufacturer {manufacturer}.\n"
    elif model_number:
        constraints_ins += f"- Match model {model_number}.\n"

    constraints_ins += (
        "- Check each chunk separately.\n"
        "- Do not skip chunks.\n"
        "- No duplicate answers.\n"
        "- Sort answers by confidence (high → low).\n"
    )

    output_ins = (
        "Output Format:\n"
        "- Format each answer as:\n"
        "<value> (<confidence>%) [Ref: <filename> page <page> line <line>]\n"
        "- Answers from 'SPECIFIC COLLECTION' are ~15% more reliable than 'SHARED'.\n"
        "- Confidence max = 100%.\n"
        "- If no answer, return: Not Found\n"
    )

    instructions = "\n---------------------\n".join([task_ins, constraints_ins, output_ins])
    developer_message = DeveloperContent.new().with_instructions(instructions)

    convo = Conversation.from_messages([
        Message.from_role_and_content(Role.SYSTEM, system_message),
        Message.from_role_and_content(Role.DEVELOPER, developer_message),
        Message.from_role_and_content(Role.USER, hits)
    ])
    return convo

def model_predict(manufacturer: str, model_number: str, query_attr: str, hits: str) -> str:
    try:
        convo = prepare_convo(manufacturer, model_number, query_attr, hits)
        prefill_ids = enc.render_conversation_for_completion(convo, Role.ASSISTANT)
        stop_token_ids = enc.stop_tokens_for_assistant_actions()

        device = next(model.parameters()).device
        input_ids = torch.tensor([prefill_ids], device=device)
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=256,
            eos_token_id=stop_token_ids,
            temperature=0
        )
        completion_ids = outputs[0][len(prefill_ids):].cpu().tolist()
        parsed = enc.parse_messages_from_completion_tokens(completion_ids, Role.ASSISTANT)

        final_msg = [msg for msg in parsed if msg.channel == "final"]
        if final_msg:
            return final_msg[-1].content[0].text
        return "No final message found"
    except Exception as e:
        return f"Something went wrong :( Error: {e}"