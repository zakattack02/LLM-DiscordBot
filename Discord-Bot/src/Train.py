import json
import os
import torch
import ijson
import psutil
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Setup device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

# Function to load the configuration from the config.json file
def load_config(config_file='Discord-Bot/src/config/config.json'):
    print(f"Loading config from {config_file}")
    if not os.path.exists(config_file):  # Check if the file exists
        raise FileNotFoundError(f"{config_file} not found. Please ensure the path is correct.")
    
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

# Function to stream texts from large NDJSON file using ijson
def load_texts_from_config(config_file='Discord-Bot/src/config/config.json'):
    print(f"Loading training data from config file...")   
    config = load_config(config_file)

    data_file_path = config.get("TrainingData", "pol_0616-1119_labeled/pol_062016-112019_labeled.ndjson")

    if not data_file_path:
        raise ValueError("Training data file path is missing in the config file.")
    
    # Check if the file exists
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"Training data file not found at the path: {data_file_path}")
    
    print(f"Streaming training data from {data_file_path}...")
    
    # Generator to stream large NDJSON file line by line
    def text_generator():
        with open(data_file_path, 'r') as file:
            for line in file:
                yield line.strip()  # Yield each line

    return text_generator()

# Function to monitor memory usage
def log_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory Usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")

# Dataset class to convert streamed texts into tokenized format
class TextDataset(Dataset):
    def __init__(self, text_generator, tokenizer, max_length=512, buffer_size=200):
        self.text_generator = text_generator  # Store the generator
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.buffer = [] 
        self.buffer_size = buffer_size  

    def _fill_buffer(self):
        """Refill buffer by pulling new data from the generator."""
        try:
            for _ in range(self.buffer_size):
                text = next(self.text_generator)
                self.buffer.append(text)
        except StopIteration:
            pass  # Stop when the generator is exhausted

    def __len__(self):
        return self.buffer_size

    def __getitem__(self, idx):
        if not self.buffer:
            self._fill_buffer()  # Fetch more data if buffer is empty
            if not self.buffer:
                raise IndexError("No more data to fetch!")  # Stop when exhausted

        text = self.buffer.pop(0)  # Retrieve from buffer
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)  # Remove batch dimension
        attention_mask = encoding["attention_mask"].squeeze(0)
        return input_ids, attention_mask

# Training function
def train_gpt_model_remote(text_generator, epochs=3, batch_size=2, lr=5e-5, accumulation_steps=4): 
    print("Starting training...")   
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)

    # Wrap generator inside the dataset class
    dataset = TextDataset(text_generator, tokenizer, max_length=512)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) 
    optimizer = AdamW(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")   
        optimizer.zero_grad()
        for step, (input_ids, attn_masks) in enumerate(dataloader):
            input_ids, attn_masks = input_ids.to(device), attn_masks.to(device)
            
            #print(f"DEBUG: Forward pass started on device {device}")   

            # Forward pass
            outputs = model(input_ids, attention_mask=attn_masks, labels=input_ids)
            loss = outputs.loss / accumulation_steps

            print(f"DEBUG: Loss Computed -> {loss.item()}")   

            # Backward pass
            loss.backward()
            
            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            print(f"Epoch: {epoch+1}, Step: {step+1}, Loss: {loss.item()}")  
            log_memory_usage()
        
        # Optionally save checkpoints after each epoch
        #save_path = f"checkpoint_epoch_{epoch+1}"
        #if not os.path.exists(save_path):
        #    os.makedirs(save_path)
        #model.save_pretrained(save_path)
        #tokenizer.save_pretrained(save_path)

    # Save the final trained model
    model.save_pretrained('trained_model')
    tokenizer.save_pretrained('trained_model')
    print("Training complete. Final model saved.") 

# Load and process the texts, then start training
try:
    text_generator = load_texts_from_config('Discord-Bot/src/config/config.json')
    print(f"Training data loaded, starting the training process...")   
    print(f"Text_Generator: {text_generator}")
    train_gpt_model_remote(text_generator)
except FileNotFoundError as e:
    print(f"Error: {e}")
except ValueError as e:
    print(f"Error: {e}")
