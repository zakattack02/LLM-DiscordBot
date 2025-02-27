import torch
import transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import ijson
import os

# Setup device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_gpt_model_remote(texts_generator, epochs=3, batch_size=2, lr=5e-5):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    
    # Create a generator to iterate through JSON file
    with open('pol_062016-112019_labeled.ndjson', 'r') as file:
        for item in ijson.items(file, 'item'):
            if item:  # Skip empty items if any
                texts.append(item['text'])
    
    # Dataset and Dataloader setup
    dataset = TextDataset(texts, tokenizer, max_length=512)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = AdamW(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(epochs):
        for batch in dataloader:
            input_ids, attn_masks = [x.to(device) for x in batch]
            outputs = model(input_ids, attention_mask=attn_masks, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"Epoch: {epoch}, Loss: {loss.item()}")
    
    # Save the trained model
    model.save_pretrained('trained_model')
    tokenizer.save_pretrained('trained_model')

# Use ijson to load JSON file in parts and generate text data
def load_texts_from_file(file_path):
    with open(file_path, 'r') as file:
        return ijson.items(file, 'item')

texts = load_texts_from_file('pol_062016-112019_labeled.ndjson')
train_gpt_model_remote(texts)