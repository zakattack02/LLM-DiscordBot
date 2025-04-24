import os
import torch
import psutil
import signal
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Setup device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Function to monitor memory usage
def log_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory Usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")

# Global variable to track manual save signal
manual_save = False  # Initialize the manual save flag

# Signal handler for manual save (Ctrl+/)
def handle_manual_save(signum, frame):
    global manual_save
    manual_save = True
    print("\033[1;33mManual save triggered (Ctrl+/ detected).\033[0m")

# Register signal handler for SIGQUIT (Ctrl+/ equivalent)
signal.signal(signal.SIGQUIT, handle_manual_save)

# Generator to load data in chunks 
def load_data_in_chunks(file_path, chunk_size=10000):
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
def train_gpt_model(file_path, epochs=2, batch_size=8, lr=5e-5, accumulation_steps=4, chunk_size=1000):
    global manual_save
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

    # Calculate total chunks and steps
    total_chunks = sum(1 for _ in load_data_in_chunks(file_path, chunk_size))
    print(f"\033[1;33mTotal chunks to process: {total_chunks}\033[0m")

    # Load data in chunks
    for epoch in range(epochs):
        print(f"\033[1;34mEpoch {epoch+1}/{epochs}\033[0m")
        for chunk_idx, data_chunk in enumerate(load_data_in_chunks(file_path, chunk_size)):
            print(f"\033[1;36mProcessing chunk {chunk_idx + 1}/{total_chunks}...\033[0m")
            dataset = TextDataset(data_chunk, tokenizer)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            total_steps = len(dataloader)
            print(f"\033[1;33mTotal steps in chunk {chunk_idx + 1}: {total_steps}\033[0m")

            for step, (input_ids, attn_masks) in enumerate(dataloader):
                input_ids, attn_masks = input_ids.to(device), attn_masks.to(device)

                # Forward pass
                outputs = model(input_ids, attention_mask=attn_masks, labels=input_ids)
                loss = outputs.loss / accumulation_steps

                print(
                    f"\033[1;34mEpoch: {epoch+1}\033[0m, "
                    f"\033[1;33mChunk: {chunk_idx+1}/{total_chunks}\033[0m, " 
                    f"\033[1;36mStep: {step+1}/{total_steps}\033[0m, "       
                    f"Loss: {loss.item():.4f}"
                )

                # Backward pass
                loss.backward()

                if (step + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                log_memory_usage()

            # Check for manual save signal
            if manual_save:
                print("\033[1;33mSaving model due to manual save request...\033[0m")
                model.save_pretrained(model_dir)
                tokenizer.save_pretrained(model_dir)
                print(f"\033[1;32mModel saved at: {model_dir}\033[0m")
                manual_save = False

        # Save the model after each epoch
        print(f"\033[1;33mSaving model after epoch {epoch+1}...\033[0m")
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)

    # Save the final trained model and tokenizer after all epochs are completed
    print("\033[1;33mSaving final model...\033[0m")
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