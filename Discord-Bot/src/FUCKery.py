import torch
import transformers
import time
import asyncio
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import sys
import os
import json
import discord
from discord import app_commands
from discord.ext import commands

import pytesseract
from PIL import Image
import requests
import io

# Personality definition
Personality = "you are Chet Gipeeti, you are an asshole, a little racist and sarcastic. Respond as such."

# Training flag
Train = False
Use_Discord = True

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


async def run_gpt_model(user_input=None):
    """ Asynchronous function to prevent blocking the event loop """
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)

    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    if not user_input:
        return "*[No input provided]*"

    # Prevent blocking
    await asyncio.sleep(0.5)

    prompt = f"{Personality}\n{user_input}"

    inputs = tokenizer.encode_plus(prompt, return_tensors='pt', padding=True, truncation=True).to(device)

    # Run model generation
    outputs = await asyncio.to_thread(
        model.generate,
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        pad_token_id=tokenizer.eos_token_id,
        max_length=inputs['input_ids'].shape[1] + 1000,
        repetition_penalty=1.2
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
    generated_text = generated_text.replace("Reply Delete", "").strip()

    return generated_text


# Load configuration
with open('C:/Users/zak/Downloads/LLM/Discord-Bot/src/config/config.json') as config_file:
    config = json.load(config_file)

TOKEN = config['bot_token']

# Initialize bot with necessary intents
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
bot = commands.Bot(command_prefix='>>', intents=intents)

# Global variables
CHANNEL_ID = None
message_counter = 0


@bot.event
async def on_ready():
    """ Called when the bot is ready. Syncs commands. """
    print(f'Logged in as {bot.user} (ID: {bot.user.id})')
    print('------')

    # Sync commands after bot is fully ready
    await bot.wait_until_ready()
    try:
        synced = await bot.tree.sync()
        print(f"✅ Synced {len(synced)} commands.")
    except Exception as e:
        print(f"❌ Failed to sync commands: {e}")

    if CHANNEL_ID:
        channel = bot.get_channel(CHANNEL_ID)
        if channel:
            print(f'Bot is monitoring channel: {channel.name} (ID: {channel.id})')
        else:
            print(f'⚠️ Channel with ID {CHANNEL_ID} not found.')
    else:
        print('⚠️ No channel ID set. Use /setchannel to set a channel.')


@bot.tree.command(name="setchannel", description="Sets the channel for the bot to monitor")
async def set_channel(interaction: discord.Interaction, channel: discord.TextChannel):
    """ Sets the channel for the bot to monitor """
    global CHANNEL_ID, message_counter

    CHANNEL_ID = channel.id
    message_counter = 0

    # Save the channel ID persistently
    with open("Discord-Bot\src\config\config.json", "w") as f:
        json.dump({"channel_id": CHANNEL_ID}, f)

    await interaction.response.send_message(f'Channel set to {channel.mention}')
    print(f'Bot is now monitoring channel: {channel.name} (ID: {channel.id})')


@bot.tree.command(name="ping", description="Responds with the bot's latency")
async def ping(interaction: discord.Interaction):
    """ Responds with the bot's latency. """
    latency = round(bot.latency * 1000, 2)
    print(f"Pong! Latency: {latency}ms")
    await interaction.response.send_message(f"Pong! Latency: {latency}ms")


@bot.tree.command(name="chat", description="Chat with the GPT-2 model")
async def chat(interaction: discord.Interaction, message: str):
    """ Responds with a message from the GPT-2 model. """
    global message_counter, CHANNEL_ID

    # ✅ Defer the response to prevent timeout
    await interaction.response.defer(thinking=True)  

    print("Thinking...")
    llm_response = await run_gpt_model(user_input=message)
    print("Chatting")

    if not llm_response.strip():
        llm_response = "*[No response generated]*"

    # ✅ Follow up with the final response after processing
    await interaction.followup.send(f"{llm_response}")

    # ✅ Manually increment `message_counter` for slash commands
    if CHANNEL_ID and interaction.channel_id == CHANNEL_ID:
        message_counter += 1
        print(f"✅ Message counter updated: {message_counter}")


@bot.tree.command(name="sync", description="Manually sync/update bot commands.")
async def sync(interaction: discord.Interaction):
    """ Manually sync/update bot commands. """
    try:
        synced = await bot.tree.sync()
        await interaction.response.send_message(f"✅ Successfully synced {len(synced)} commands.", ephemeral=True)
        print(f"✅ Manually synced {len(synced)} commands.")
    except Exception as e:
        await interaction.response.send_message(f"❌ Failed to sync commands: {e}", ephemeral=True)
        print(f"❌ Failed to sync commands: {e}")



@bot.event
async def on_message(message):
    """ Processes messages and responds if needed. """
    global CHANNEL_ID, message_counter

    if message.author == bot.user:
        return

    content = message.content

    # Handle Tenor GIFs
    for embed in message.embeds:
        if "tenor.com" in embed.url:
            content += f"\n[Tenor GIF]: {embed.url}"

    print(f'📨 Message {message_counter} in channel: {content}')

    if CHANNEL_ID and message.channel.id == CHANNEL_ID:
        message_counter += 1  # ✅ Now this will increment properly
        print(f"✅ Message counter updated: {message_counter}")

        if message_counter % 3 == 0:
            print('Thinking...')
            llm_response = await run_gpt_model(user_input=content)
            print('Chatting')

            if not llm_response.strip():
                llm_response = "*[No response generated]*"

            await message.channel.send(f"{llm_response}")

    await bot.process_commands(message)  # ✅ Fixed missing await


# Run terminal chat
async def run_terminal_chat():
    print("GPT Model Chat - Type 'exit' to quit")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        response = await run_gpt_model(user_input)
        print(f"Chet: {response}\n")


# Start either Discord bot or terminal chat
if Use_Discord:
    bot.run(TOKEN)
else:
    asyncio.run(run_terminal_chat())
