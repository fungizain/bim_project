from typing import Dict
from uuid import uuid4
import pprint

SYSTEM_PROMPT = """You are a retrieval-augmented assistant. Follow the rules strictly:
- Extract the most relevant values for the target attribute.
- Match Manufacturer and Model Number if provided.
- Return up to 3 candidate answers.
- Do NOT add explanations or commentary.
"""

PROMPT_TEMPLATE = {
    "messages": [
        {
            "role": "developer",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": (
                "context:\n{hits}\n\n"
                "manufacturer: {manufacturer}\n"
                "model number: {model_number}\n"
                "query_attr: {query_attr}"
            )
        },
        {
            "role": "assistant",
            "content": (
                "We need to answer in the following format:\n"
                "<value> (<confidence>%) [Reference: <source.pdf> page <page> line <line>]"
            )
        }
    ]
}

_session_prompts : Dict[str, str] = {}

def create_session() -> str:
    session_id = str(uuid4())
    _session_prompts[session_id] = SYSTEM_PROMPT
    print(f"[DEBUG] Created new session: {session_id}")
    print(f"[DEBUG] Initial prompt template set:\n{_session_prompts[session_id]}")
    return session_id

def change_system_template(session_id: str, new_prompt: str) -> None:
    _session_prompts[session_id] = new_prompt
    print(f"[DEBUG] Changed SYSTEM_PROMPT for session {session_id}")
    print(f"[DEBUG] New SYSTEM_PROMPT:\n{new_prompt}")

def get_system_prompt(session_id: str) -> str:
    prompt = _session_prompts.get(session_id, SYSTEM_PROMPT)
    print(f"[DEBUG] Retrieved SYSTEM_PROMPT for session {session_id}")
    return prompt

def prepare_prompt_with_template(
    manufacturer: str,
    model_number: str,
    query_attr: str,
    hits: str,
    system_prompt: str = None
) -> str:
    system_message = system_prompt or SYSTEM_PROMPT
    user_message = PROMPT_TEMPLATE["messages"][1]["content"].format(
        manufacturer=manufacturer,
        model_number=model_number,
        query_attr=query_attr,
        hits=hits
    )
    messages = [
        f"DEVELOPER:\n{system_message}",
        f"USER:\n{user_message}",
        f"ASSISTANT (format example):\n{PROMPT_TEMPLATE['messages'][2]['content']}"
    ]
    print(f"[DEBUG] Harmony Prompt (system):\n{system_message}")
    prompt_str = "\n\n".join(messages)
    return prompt_str

def prepare_prompt(
        session_id: str,
        manufacturer: str,
        model_number: str,
        query_attr: str,
        hits: str
    ) -> dict:
    template = get_system_prompt(session_id)
    print(f"[DEBUG] Prepared prompt for session {session_id}")
    prompt = prepare_prompt_with_template(
        manufacturer=manufacturer,
        model_number=model_number,
        query_attr=query_attr,
        hits=hits,
        system_prompt=template
    )
    return prompt
