import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class CustomLLMInference:
    def __init__(self, model_name_or_path):
        # Load the tokenizer and model from Huggingface or local path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    
    def predict(self, input_text):
        # Tokenize input text
        inputs = self.tokenizer(input_text, return_tensors="pt")
        
        # Perform generation
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs)
        
        # Decode the generated tokens back to text
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_text
