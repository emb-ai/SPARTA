from transformers import AutoModelForCausalLM, AutoTokenizer

def download_model():
    model_name = "xinlai/LISA-13B-llama2-v0-explanatory"
    
    print(f"Downloading model {model_name}...")
    
    # Download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Download model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    
    print("Model downloaded successfully!")

if __name__ == "__main__":
    download_model() 