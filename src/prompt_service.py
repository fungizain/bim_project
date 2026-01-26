from typing import Dict
from uuid import uuid4

PROMPT_TEMPLATE = """
You are a retrieval-augmented assistant.
Follow the rules strictly:
- Extract the most relevant values for the target attribute.
- Match Manufacturer and Model Number if provided.
- Return up to 3 candidate answers.
- Do NOT add explanations or commentary.
- Format strictly as: <value> (<confidence>%) [Reference: <source.pdf> page <page> line <line>]

---

context:
{hits}

manufacturer: {manufacturer}
model number: {model_number}
query_attr: {query_attr}
"""

_session_prompts : Dict[str, str] = {}

def create_session() -> str:
    session_id = str(uuid4())
    _session_prompts[session_id] = PROMPT_TEMPLATE
    print(f"[DEBUG] Created new session: {session_id}")
    print(f"[DEBUG] Initial prompt template set:\n{_session_prompts[session_id]}")
    return session_id

def change_prompt_template(session_id: str, new_template: str) -> None:
    _session_prompts[session_id] = new_template
    print(f"[DEBUG] Changed prompt template for session {session_id}")
    print(f"[DEBUG] New template:\n{new_template}")

def get_prompt_template(session_id: str) -> str:
    template = _session_prompts.get(session_id, PROMPT_TEMPLATE)
    print(f"[DEBUG] Retrieved prompt template for session {session_id}")
    return template

def prepare_prompt_with_template(
    manufacturer: str,
    model_number: str,
    query_attr: str,
    hits: str,
    prompt_template: str = None
) -> str:
    template = prompt_template or PROMPT_TEMPLATE
    prompt = template.format(
        manufacturer=manufacturer,
        model_number=model_number,
        query_attr=query_attr,
        hits=hits
    )
    print(f"[DEBUG] Prompt:\n{prompt.split('---')[0].strip()}")
    return prompt 

def prepare_prompt(
        session_id: str,
        manufacturer: str,
        model_number: str,
        query_attr: str,
        hits: str
    ) -> str:
    template = get_prompt_template(session_id)
    prompt = prepare_prompt_with_template(
        manufacturer=manufacturer,
        model_number=model_number,
        query_attr=query_attr,
        hits=hits,
        prompt_template=template
    )
    print(f"[DEBUG] Prepared prompt for session {session_id}")
    return prompt
