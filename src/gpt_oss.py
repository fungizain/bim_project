from transformers import pipeline, AutoTokenizer

model_id = "openai/gpt-oss-20b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="auto",
    device_map="auto"
)

messages = [
    {"role": "user", "content": "Tell me a joke."}
]

chat_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(chat_input)

outputs = pipe(messages, max_new_tokens=1024)

full_text = tokenizer.apply_chat_template(outputs[0]["generated_text"], tokenize=False)
print(full_text)