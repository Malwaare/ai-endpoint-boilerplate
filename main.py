import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from fastapi import FastAPI

app = FastApi()


@app.get("/ai/api/your-endpoint")
async def function():
    # load model with tokenizer, replace your model name
    model = AutoModel.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True)
    # Save them locally if you want
    model.save_pretrained("./model")

    # get the embeddings
    max_length = 32768
    query_embeddings = model.encode(queries, instruction=query_prefix, max_length=max_length)
    passage_embeddings = model.encode(passages, instruction=passage_prefix, max_length=max_length)

    # normalize embeddings
    query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
    passage_embeddings = F.normalize(passage_embeddings, p=2, dim=1)

    scores = (query_embeddings @ passage_embeddings.T) * 100
    return scores.tolist()