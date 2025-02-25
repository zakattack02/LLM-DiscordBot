import os
import time
import subprocess

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

BOT_SCRIPT = "Discord-Bot\src\FUCKery.py"  # Change to your bot’s script filename

while True:
    print("🔄 Starting bot...")
    process = subprocess.Popen(["python", BOT_SCRIPT])

    process.wait()

    if os.path.exists("restart.flag"):
        print("🔁 Restarting bot...")
        os.remove("restart.flag")  
        time.sleep(3)  
    else:
        print("❌ Bot stopped manually. Exiting restart script.")
        break
