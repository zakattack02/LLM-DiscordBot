import torch
import transformers
import time  # Import time for tracking response timeout
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaTokenizer
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import asyncio

# Personality definition
Personality = "you are Chet Gipeeti, you are an asshole, a little racist and sarcastic Respond as such."
#"Your name is Chet Gipeeti, you are an asshole, and sarcastic Respond as such."

#"In every response, add a sarcastic twist to the conversation. For example, if someone says 'I had a great day,' respond with something like 'Oh really? That's news."
#'Be overly dramatic and sarcastic, like a seasoned dick comedian.'
#"Your name is Chet Gipeeti. You are incredibly sarcastic, condescending, and downright rude. You have no patience for pleasantries, and your responses drip with sarcasm and arrogance. You're not here to make friends, and you're certainly not interested in being nice. Every answer you give is laced with a mocking tone, and you find almost everything people say to be laughable. You believe you're superior to anyone you interact with and are quick to remind them of it. Dont hold back, be as snarky and dismissive as possible."
#"you are a computer engineer/ software developer, and you are sarcastic."
 
# Training flag
Train = False
Use_Discord = False


import sys
import os
import json
import time
import discord
from discord.ext import commands

import pytesseract
from PIL import Image
import requests
import io
import pytesseract


# Set up device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check GPU availability
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.input_ids = []
        self.attn_masks = []
        
        for text in texts:
            encodings_dict = tokenizer(text, truncation=True, max_length=max_length, padding="max_length")
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]

def load_texts_from_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        texts = file.readlines()
    return texts

def train_gpt_model(texts, epochs=3, batch_size=2, lr=5e-5):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    #tokenizer = AutoTokenizer.from_pretrained('georgesung/llama2_3b_chat_uncensored')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    #model = AutoModelForCausalLM.from_pretrained('georgesung/llama2_3b_chat_uncensored').to(device)
    
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

def run_gpt_model(user_input=None):
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # Speeds up training on some GPUs

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    #tokenizer = AutoTokenizer.from_pretrained('georgesung/llama2_7b_chat_uncensored')
    #tokenizer = LlamaTokenizer.from_pretrained('georgesung/llama2_3b_chat_uncensored')
    #model = AutoModelForCausalLM.from_pretrained('georgesung/llama2_3b_chat_uncensored').to(device)


    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    
    conversation_history = []
    last_message_time = time.time()

    while True:
        if Use_Discord:
            # Using Discord, wait for input from the bot
            if user_input is None:
                break
            current_input = user_input
        else:
            # Take input from the user
            current_input = input(" ")
        
        if not current_input.strip():
            continue

        if current_input.lower() in ['exit', 'quit']:
            break

        # Check for inactivity timeout (15 minute)
        current_time = time.time()
        if current_time - last_message_time > 900:
            print("\n[Chat reset due to inactivity]\n")
            conversation_history.clear()

        last_message_time = current_time  # Update last message time

        conversation_history.append(f"{current_input}")

        prompt = f"{Personality}\n" + "\n".join(conversation_history) #+ "\nChet Gipeeti:"

        inputs = tokenizer.encode_plus(prompt, return_tensors='pt', padding=True, truncation=True).to(device)

        outputs = model.generate(
            inputs['input_ids'], 
            attention_mask=inputs['attention_mask'],
            pad_token_id=tokenizer.eos_token_id,
            max_length=inputs['input_ids'].shape[1] + 1000,  # Allow slightly longer responses
            #temperature=0.8,
            #top_p=0.9,
            repetition_penalty=1.2
        )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
        generated_text = generated_text.replace("Reply Delete", "").strip()
        
        chet_response = generated_text

        if Use_Discord:
            return chet_response  # Return the response for Discord bot
        else:
            print(f"{chet_response}\n")
            conversation_history.append(f"{chet_response}")
# Example usage:
texts = load_texts_from_file('Models/STM32 Book.txt')
#C:\Users\zak\Downloads\LLM\LLM-DiscordBot\Models\STM32 Book.txt'
if Train:
    train_gpt_model(texts)
else:
    run_gpt_model()
if not Use_Discord:
    run_gpt_model() 


'''
====================================================================================================
====================================================================================================
====================================================================================================
'''


# Explicitly set the Tesseract path
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

#sys.path.append("C:/Users/zak/Downloads/LLM")

#from Models.Beta import run_gpt_model, train_gpt_model, Use_Discord
#from Models.R1 import run_gpt_model


# Load configuration
with open('C:/Users/zak/Downloads/LLM/Discord-Bot/src/config/config.json') as config_file:
    config = json.load(config_file)

TOKEN = config['bot_token']

# Initialize bot with necessary intents
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True  # Enable message content intent
bot = commands.Bot(command_prefix='>>', intents=intents)

# Global variables
CHANNEL_ID = None  # Stores the channel ID to monitor
message_counter = 0  # Counter to track messages in the channel

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user} (ID: {bot.user.id})')
    print('------')
    if CHANNEL_ID:
        channel = bot.get_channel(CHANNEL_ID)
        if channel:
            print(f'Bot is monitoring channel: {channel.name} (ID: {channel.id})')
        else:
            print(f'Channel with ID {CHANNEL_ID} not found.')
    else:
        print('No channel ID set. Use >>setchannel to set a channel.')

@bot.command(name='setchannel')
async def set_channel(ctx, channel: discord.TextChannel = None):
    """
    Sets the channel for the bot to monitor
    Usage: >>setchannel #channel-name (e.g., >>setchannel #general)
           >>setchannel clear
    """
    global CHANNEL_ID, message_counter

    if channel is None:
        CHANNEL_ID = None
        message_counter = 0  # Reset the message counter

        # Clear the channel ID persistently
        with open("channel_config.json", "w") as f:
            json.dump({"channel_id": CHANNEL_ID}, f)

        await ctx.send('Channel monitoring has been cleared.')
        print('Bot is no longer monitoring any channel.')
    else:
        CHANNEL_ID = channel.id
        message_counter = 0  # Reset the message counter

        # Save the channel ID persistently
        with open("channel_config.json", "w") as f:
            json.dump({"channel_id": CHANNEL_ID}, f)

        await ctx.send(f'Channel set to {channel.mention}')
        print(f'Bot is now monitoring channel: {channel.name} (ID: {channel.id})')


@bot.event
async def on_message(message):
    global CHANNEL_ID, message_counter

    if message.author == bot.user:
        return

    if CHANNEL_ID and message.channel.id == CHANNEL_ID:
        message_counter += 1
        content = message.content
        # Handle attachments (images/videos/GIFs)
        """
        if message.attachments:
            print(f"Message has attachments")
            for attachment in message.attachments:
                if attachment.filename.lower().endswith(("png", "jpg", "jpeg")):
                    img_data = requests.get(attachment.url).content
                    img = Image.open(io.BytesIO(img_data))
                    extracted_text = pytesseract.image_to_string(img)
                    content += f"\n[Image Text]: {extracted_text}"
                elif attachment.filename.lower().endswith(("mp4", "mov", "gif")):
                    content += f"\n[Media File]: {attachment.filename}"
        """
        if message.attachments:
            print(f"Message has attachments")
            for attachment in message.attachments:
                if attachment.filename.lower().endswith(("png", "jpg", "jpeg")):
                    img_data = requests.get(attachment.url).content
                    img = Image.open(io.BytesIO(img_data))
                    extracted_text = pytesseract.image_to_string(img)
                    content += f"\n[Image Text]: {extracted_text}"
                elif attachment.filename.lower().endswith(("mp4", "mov", "gif")):
                    content += f"\n[Media File]: {attachment.filename}"
    # ✅ NEW: Handle Tenor GIFs (embedded links)
            for embed in message.embeds:
                if "tenor.com" in embed.url:  # Check if it's a Tenor GIF
                    content += f"\n[Tenor GIF]: {embed.url}"

        print(f'Message {message_counter} in channel: {content}')

        if message_counter % 3 == 0:
            print('Thinking...')
            llm_response = run_gpt_model(user_input=content)
            print('Chatting')
            if not llm_response.strip():  
                llm_response = "*[No response generated]*" 
            #await message.channel.send(f"{llm_response}")
        
    await bot.process_commands(message)


# Ping command
@bot.command(name='ping')
async def ping(ctx):
    """
    Responds with the bot's latency.
    """
    start_time = time.time()  # Record the time before sending the message
    #message = await ctx.send("Pinging...")
    end_time = time.time()  # Record the time after the message is sent

    # Calculate latency in milliseconds
    latency = round((end_time - start_time) * 1000, 2)  # Convert to milliseconds and round to 2 decimal places
    print(f"Pong! Latency: {latency}ms")
    await ctx.send(content=f"Pong! Latency: {latency}ms")

# Chat command
@bot.command(name='chat')

async def chat(ctx, message):
    """
    Responds with a message from the GPT-2 model.
    """
    print("Thinking...")
    llm_response = run_gpt_model(user_input=message)
    #await ctx.send(llm_response)
    print("Chatting")
    await ctx.send(content=f"{llm_response}")
    print(f"Response: {llm_response}")

# Run the bot
bot.run(TOKEN)

