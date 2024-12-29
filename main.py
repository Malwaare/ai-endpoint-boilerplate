import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from fastapi import FastAPI

app = FastApi()


@app.get("/ai/api/your-endpoint")
async def your_endpoint(input: str):
    # load model with tokenizer, replace your model name
    model = AutoModel.from_pretrained('google/gemma-2-2b-it', trust_remote_code=True)
    # Save them locally if you want
    model.save_pretrained("./model")

    #Gemma example
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b-it",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    input_text = input
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

    outputs = model.generate(**input_ids, max_new_tokens=32)

    return tokenizer.decode(outputs[0])