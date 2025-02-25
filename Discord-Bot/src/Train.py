import torch
import transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

# Setup device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training function (run remotely on DigitalOcean)
def train_gpt_model_remote(texts, epochs=3, batch_size=2, lr=5e-5):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    
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

    # Optionally upload back to cloud storage
    

# Load training data and trigger training
texts = load_texts_from_file('training_data.txt')
train_gpt_model_remote(texts)
