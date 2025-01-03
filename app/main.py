import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI
from contextlib import asynccontextmanager
import gc

# Global variables to store model and tokenizer
model = None
tokenizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model and tokenizer when the application starts
    global model, tokenizer
    print("Loading model and tokenizer...")
    
    # Clear cache and garbage collect
    gc.collect()
    
    # Check if MPS is available (Macs)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        "google/gemma-2-2b-it",
        padding_side="left",
        truncation_side="left"
    )
    # Ensure pad token is set correctly
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Configure memory settings
    max_memory = {
        "cpu": "16GB",  # Adjust based on your MacBook's RAM
        "mps": "16GB"   # Adjust based on your M2 Pro's memory
    }
    
    # Load model with Apple Silicon optimizations
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b-it",
        torch_dtype=torch.float16,  # M2 Pro supports float16
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory=max_memory,
        offload_folder="offload_folder"
    )
    
    print("Model and tokenizer loaded successfully")
    yield
    # Clean up resources
    print("Shutting down the application")
    gc.collect()

app = FastAPI(lifespan=lifespan)

@app.get("/ai/api/your-endpoint")
async def your_endpoint(input: str):
    try:
        # Create a simple, consistent prompt
        prompt = f"""Answer the following question accurately and concisely.
        Question: {input}
        Answer: Let me provide a clear answer."""
        
        # Tokenize with proper padding and attention mask
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        
        # Ensure consistent device placement
        device = next(model.parameters()).clone().detach().device
        input_ids = inputs["input_ids"].clone().detach().to(device)
        attention_mask = inputs["attention_mask"].clone().detach().to(device)
        
        # Generate with careful parameter settings
        with torch.inference_mode():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=50,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
            )
        
        response = tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).strip()
        
        return response
    
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)