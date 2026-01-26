import json
import torch
from transformers import pipeline, AutoTokenizer, GptOssForCausalLM
from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    Role,
    SystemContent,
    ToolDescription,
    load_harmony_encoding,
    ReasoningEffort
)

context = """[chunk ad9e211ed33ca8ab0a4426fb7f22815f | distance: 0.8613]
source: Testing File Simple.pdf | pages: 1-3
Sample Operation and Maintenance Manual
For LF1101 Electric Lifts
Venue: EMSDN
Asset Code: KT-EMSDN-NA-000-LAE-ELL-0015
Page 1 of 3
Common Attribute
### Table 1
- **Acquisition Date**:
- **Acquisition Value**:
- **Asset Code**: Attribute Values=KT-EMSDN-NA-000-LAE-ELL-0015
- **Asset Relationship**:
- **Asset Tag No**:
- **Authorization Group**: Attribute Values=GEOO
- **CCS Superior Equipment No**:
- **CCS Superior Equipment Technical ID**:
- **No**:
- **Cover Photo**:
- **Currency**:
- **Customer Warranty End**:
- **Customer Warranty Start**:
- **Division**: Attribute Values=02
- **Documentation**:
- **Equipment Description**: Attribute Values=PASS, L#9, 900kg, 1.60m / s, 8 ENT
- **Equipment Long Text**:
- **Equipment No**: Attribute Values=10829923
- **Functional Location**: Attribute Values=EMSDN-LF
- **Grouped Equipment ID**:
- **Inventory No**:
- **Main Work Centre**: Attribute Values=GK2A6H60
- **Manufacturer**: Attribute Values=fujitec
- **Manufacturer Country Or Region**: Attribute Values=japan
- **Manufacturer Serial No**:
- **Mark Delete**:
- **Model No**: Attribute Values=EXDN
- **Partner ID**: Attribute Values=EMSTF
- **Planner Group**: Attribute Values=Goo
- **Room / Floor**:
Start-up Date
Technical ID No.
User Status (BER)
Page 2 of 3
### Table 1
- **Row 1**: User Status (BOS)=User Status (LTS)
- **Row 2**: User Status (BOS)=User Status (SER)
- **Row 3**: User Status (BOS)=User Status (SUS)
- **Row 4**: User Status (BOS)=Vendor Warranty End
- **Row 5**: User Status (BOS)=Vendor Warranty Start
- **Row 6**: User Status (BOS)=Weight
- **Row 7**: User Status (BOS)=Zone Tag No
Specific Attribute
Attribute Name
Attribute Values
### Table 1
- **Row 1**: Column 3=Car Floor Area (m2)
- **Construction of Suspension Rope**:
- **Control**:
- **Date of Last Suspension Rope**:
- **Replacement**:
- **Door Type**: Passenger Lift Fujitec=HORIZONTAL CENTRE OPENING
- **Fireman's Lift**:
- **Length of Travel (m)**: Passenger Lift Fujitec=43.76
- **Levels Served**: Passenger Lift Fujitec=8
- **Lift No**: Passenger Lift Fujitec=L9
- **Location (Address) on Use Permit**: Passenger Lift Fujitec=EMSD HQS, 3 Kai Shing Sheet, Kowloon Bay, Kowloon
- **Location ID on Use Permit**:
- **Machine Room Location**: Passenger Lift Fujitec=AT SIDE
- **Model**:
- **Motor Rating (kW)**:
- **No. of Suspension Rope**:
- **Nominal Diameter of Suspension Rope (mm)**:
- **Rated Load**: Passenger Lift Fujitec=630
- **Rated Speed**: Passenger Lift Fujitec=0.5
- **Type of Drive**:
Year of Installation
2004
Page 3 of 3"""

encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
system_message = (
    SystemContent.new()
        .with_model_identity(
            "You are a ChatGPT, a large language model trained by OpenAI."
        )
        .with_reasoning_effort(ReasoningEffort.MEDIUM)
        .with_conversation_start_date("2026-01-26")
        .with_knowledge_cutoff("2024-06")
        .with_required_channels(["final"])
)
developer_message = (
    DeveloperContent.new()
        .with_instructions("Answer in pirate speak")
)
convo = Conversation.from_messages([
    Message.from_role_and_content(Role.SYSTEM, system_message),
    Message.from_role_and_content(Role.USER, "hello")
])
prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
stop_token_ids = encoding.stop_tokens_for_assistant_actions()

model_id = "openai/gpt-oss-20b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = GptOssForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    dtype="auto"
)

device = next(model.parameters()).device
input_ids = torch.tensor([prefill_ids], device=device)

outputs = model.generate(
    input_ids=input_ids,
    max_new_tokens=1024,
    eos_token_id=stop_token_ids
)

completion_ids = outputs[0][len(prefill_ids):].cpu().tolist()

displayed_ids = []
for token_id in completion_ids:
    if token_id in stop_token_ids:
        displayed_ids.append(token_id)
        break
    displayed_ids.append(token_id)

parse_ids = [tid for tid in displayed_ids if tid not in stop_token_ids]

try:
    entries = encoding.parse_messages_from_completion_tokens(parse_ids, Role.ASSISTANT)
    
    print("\nPARSED MESSAGES:")
    print("-" * 40)
    for message in entries:
        print(json.dumps(message.to_dict(), indent=2))
        print("-" * 40)
except Exception as e:
    print(f"\nError parsing messages: {e}")
    print("\nRaw decoded output:")
    print(encoding.decode(parse_ids))