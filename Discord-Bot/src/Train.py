import json
import os
import sys
import torch
import ijson
import psutil
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Setup device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text_byte=0 

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

    data_file_path = config.get("TrainingData")

    if not data_file_path:
        raise ValueError("Training data file path is missing in the config file.")
    
    # Check if the file exists
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"Training data file not found at the path: {data_file_path}")
    
    print(f"Streaming training data from {data_file_path}...")

    # Generator to stream approximately 6 GiB worth of lines at a time
    def text_generator():
        #Whole file is 106 GiB
        chunk_size =  30* 1024 ** 2  # ** GiB in bytes
        current_chunk = []
        current_size = 0

        with open(data_file_path, 'r') as file:
            for line in file:
                line_size = len(line.encode('utf-8'))  # Calculate size of the line in bytes
                current_chunk.append(line.strip())
                current_size += line_size

                if current_size >= chunk_size: # If the chunk size is reached
                    yield current_chunk
                    current_chunk = []  # Reset the chunk
                    current_size = 0

            # Yield any remaining lines in the last chunk
            if current_chunk:
                yield current_chunk

    return text_generator()

# Function to monitor memory usage
def log_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory Usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")

# Dataset class to convert streamed texts into tokenized format
class TextDataset(Dataset):
    def __init__(self, text_generator, tokenizer, max_length=1024, batch_size=500, num_workers=2):
        self.text_generator = text_generator  # Store the generator
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.buffer = []  # Buffer
        self.buffer_size = batch_size*num_workers

    def _fill_buffer(self):
        """Refill buffer by pulling new data from the generator."""
        try:
            while len(self.buffer) < self.buffer_size:  # Ensure buffer is filled to buffer size
                chunk = next(self.text_generator)  # Get the next chunk
                print(f"Fetched a new chunk with {len(chunk)} lines.")  # Debug
                self.buffer.extend(chunk)  # Add the chunk
            print(f"Buffer size after refill: {len(self.buffer)}")  # Debug
        except StopIteration:
            print("No more chunks to fetch.")  # Debug

    def __len__(self):
        return self.buffer_size

    def __getitem__(self, idx):
        if not self.buffer:
            self._fill_buffer()
            if not self.buffer:
                raise IndexError("No more data to fetch!")

        text = self.buffer.pop(0)
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        return input_ids, attention_mask

# Training function
def train_gpt_model_remote(text_generator, epochs=8, batch_size=2, lr=5e-5, accumulation_steps=4, num_workers=5): 
    print("Starting training...")

    # Check if a pretrained model exists
    model_dir = os.path.abspath('trained_model') 
    if os.path.exists(model_dir):
        print(f"Loading pretrained model and tokenizer from {model_dir}...")
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        model = GPT2LMHeadModel.from_pretrained(model_dir).to(device)
    else:
        print("No pretrained model found. Initializing a new model...")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token
        model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)

    # Wrap generator inside the dataset class
    dataset = TextDataset(text_generator, tokenizer, max_length=1024)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers) 

    # Calculate steps per epoch after initializing the dataloader
    steps_per_epoch = len(dataloader)
    print(f"\033[1;33mSteps per epoch: {steps_per_epoch}\033[0m") 

    optimizer = AdamW(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        print(f"\033[1;34mEpoch {epoch+1}/{epochs}\033[0m")
        optimizer.zero_grad()
        for step, (input_ids, attn_masks) in enumerate(dataloader):
            input_ids, attn_masks = input_ids.to(device), attn_masks.to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask=attn_masks, labels=input_ids)
            loss = outputs.loss / accumulation_steps

            print(f"\033[1;34mEpoch: {epoch+1}\033[0m, Step: {step+1}, Loss: {loss.item():.4f}")  

            # Backward pass
            loss.backward()

            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            log_memory_usage()

    # Save the final trained model and tokenizer after all epochs are completed
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"\033[1;32m\033[1mTRAINING COMPLETE. FINAL MODEL AND TOKENIZER SAVED AT: {model_dir}\033[0m")  # Bold Green

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
