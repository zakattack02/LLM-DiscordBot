import os
import torch
import psutil
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Setup device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to monitor memory usage
def log_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory Usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")

# Generator to load data in chunks from gpt2_ready_dataset.txt
def load_data_in_chunks(file_path, chunk_size=1000):
    """
    Reads the file line by line and yields chunks of lines.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        chunk = []
        for line in file:
            if line.strip():  # Skip empty lines
                chunk.append(line.strip())
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:  # Yield any remaining lines
            yield chunk

# Dataset class to handle chunks of data
class TextDataset(Dataset):
    def __init__(self, data_chunk, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = data_chunk

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
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
def train_gpt_model(file_path, epochs=2, batch_size=3, lr=5e-5, accumulation_steps=4, chunk_size=1000):
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

    optimizer = AdamW(model.parameters(), lr=lr)
    model.train()

    # Load data in chunks
    for epoch in range(epochs):
        print(f"\033[1;34mEpoch {epoch+1}/{epochs}\033[0m")
        for chunk_idx, data_chunk in enumerate(load_data_in_chunks(file_path, chunk_size)):
            print(f"Processing chunk {chunk_idx + 1}...")
            dataset = TextDataset(data_chunk, tokenizer)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            for step, (input_ids, attn_masks) in enumerate(dataloader):
                input_ids, attn_masks = input_ids.to(device), attn_masks.to(device)

                # Forward pass
                outputs = model(input_ids, attention_mask=attn_masks, labels=input_ids)
                loss = outputs.loss / accumulation_steps

                print(f"\033[1;34mEpoch: {epoch+1}\033[0m, Chunk: {chunk_idx+1}, Step: {step+1}, Loss: {loss.item():.4f}")

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

# Main script
if __name__ == "__main__":
    file_path = "gpt2_ready_dataset.txt"  # Path to the preprocessed dataset
    try:
        train_gpt_model(file_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")