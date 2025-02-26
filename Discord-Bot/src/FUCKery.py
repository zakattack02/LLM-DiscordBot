import torch
import transformers
import time
import asyncio
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BlipProcessor, BlipForConditionalGeneration
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import sys
import os
import json
import discord
from discord import app_commands
from discord.ext import commands
from PIL import Image, ImageSequence
import requests
import io
from ultralytics import YOLO
import cv2
import numpy as np
import aiohttp
import re
import imageio_ffmpeg as ffmpeg
import ffmpeg

import pytesseract
# Explicitly set the Tesseract path
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"


# Load YOLOv8 model (pretrained on COCO dataset)
yolo_model = YOLO("yolov8n.pt")


# Personality definition
Personality = "you are Chet Gipeeti, you are an asshole, a little racist and sarcastic. Respond as such, with emojis when appropriate."

# Flags
with open('Discord-Bot/src/config/config.json') as config_file:
    config = json.load(config_file)
    Train = config["Training"]
    Use_Discord = config["Use_Discord"]

# Global variables
TOKEN = config_file['bot_token']
global CHANNEL_ID
CHANNEL_ID = None
message_counter = 0
recent_messages = []  # Store recent messages



# Set up device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check GPU availability
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# BLIP-2 for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

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
        max_length=inputs['input_ids'].shape[1] + 1008,
        repetition_penalty=1.2
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
    generated_text = generated_text.replace("Reply Delete", "").strip()

    return generated_text

async def get_tenor_gif_url(tenor_url):
    """ Extracts the direct GIF URL from a Tenor link. """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(tenor_url) as response:
                if response.status == 200:
                    html = await response.text()

                    # Extract direct GIF URL from the HTML response
                    gif_url_match = re.search(r'"(https://media\.tenor\.com/[^"]+\.gif)"', html)
                    if gif_url_match:
                        return gif_url_match.group(1)

    except Exception as e:
        print(f"Failed to extract Tenor GIF: {e}")
    return None

# Handle Animated GIFs by extracting the first frame
def get_first_frame(image):
    """ Extracts the first frame from an animated GIF or returns the image itself if static. """
    if hasattr(image, "is_animated") and image.is_animated:
        frame = next(ImageSequence.Iterator(image)) 
        return frame.convert("RGB")  # Convert to standard format
    return image.convert("RGB") 

# OCR (Text Extraction)
def preprocess_image_for_ocr(image):
    """ Convert image to grayscale and apply thresholding for better OCR results. """
    image = get_first_frame(image)

    img_array = np.array(image)

    if len(img_array.shape) == 2:
        gray = img_array
    elif img_array.shape[2] == 4:  # RGBA -> RGB -> Grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
    elif img_array.shape[2] == 3:  # RGB -> Grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError(f"Unexpected image format: {img_array.shape}")

    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(thresh)

def extract_text_from_image(image):
    """ Extracts text from an image using Tesseract OCR. """
    preprocessed_img = preprocess_image_for_ocr(image)
    return pytesseract.image_to_string(preprocessed_img).strip()

# Image Captioning (BLIP-2)
def generate_image_caption(image):
    """ Generates an image caption using BLIP-2. """
    image = get_first_frame(image)  

    inputs = processor(image, return_tensors="pt").to(device)
    output = caption_model.generate(**inputs)
    return processor.decode(output[0], skip_special_tokens=True)

# Object Detection (YOLOv8)
def detect_objects_with_yolo(image):
    """ Runs YOLOv8 on an image and extracts detected objects with confidence scores. """
    image = get_first_frame(image) 

    img_array = np.array(image)
    
    # Convert RGBA to RGB
    if img_array.shape[-1] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

    results = yolo_model(img_array)

    if not results[0].boxes:
        return ["No objects detected"]

    detected_objects = [
        f"{yolo_model.names[int(box.cls)]} ({float(box.conf):.2f})"
        for box in results[0].boxes
    ]
    return detected_objects

# Video Processing
def extract_video_thumbnail(video_url=None, local_path=None):
    """ Extracts a thumbnail from a video using FFmpeg. """
    os.environ["PATH"] += os.pathsep + "C:/ffmpeg/bin"
    temp_video = "temp_video.mp4" if video_url else local_path
    output_file = "thumbnail.jpg"

    try:
        # Download video
        if video_url:
            response = requests.get(video_url, stream=True)
            with open(temp_video, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        # Extract Thumbnail (1-second mark)
        (
            ffmpeg
            .input(temp_video, ss=1)
            .output(output_file, vframes=1)
            .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
        )

        # Remove temp video
        if video_url:
            os.remove(temp_video)

        return output_file
    except Exception as e:
        print(f"Failed to extract thumbnail: {e}")
        return None


# Load configuration
with open('Discord-Bot\src\config\config.json') as config_file:
    config_file = json.load(config_file)


# Initialize bot with necessary intents
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
bot = commands.Bot(command_prefix=config_file['prefix'], intents=intents)



@bot.event
async def on_ready():
    """ Called when the bot is ready. Syncs commands. """
    #print("Config file contents:", config_file)
    await bot.tree.sync()
    print(f'synced {len(bot.commands)} commands')

    config_file["channel_id"] = None
    print(f'Logged in as {bot.user} (ID: {bot.user.id})')
    print(f'Command prefix is: {config_file["prefix"]}')
    print(f'CHANNEL_ID: {config_file["channel_id"]}')
    print(f'Reboot channel: {config_file["reboot_channel"]}')
    print('------')


    with open("Discord-Bot/src/config/config.json", "r") as f:
        config = json.load(f)
    reboot_channel_id = config.get("reboot_channel", None)

    if reboot_channel_id:
        try:
            channel = bot.get_channel(int(reboot_channel_id))
            if channel:
                await channel.send("✅ **I'm back online!** ")
                #llm_message = await run_gpt_model(user_input="Give me a message that you're back online")
                #await channel.send(llm_message)
                print(f"✅ Sent reboot message to {channel.name} (ID: {reboot_channel_id})")
            else:
                print("⚠️ Reboot channel not found.")
        except Exception as e:
            print(f"❌ Failed to send reboot message: {e}")

        # Clear reboot_channel
        config["reboot_channel"] = None
        config["channel_id"] = None
        with open("Discord-Bot/src/config/config.json", "w") as f:
            json.dump(config, f, indent=4)


async def no_longer_monitoring_message():
    """ Generates a message indicating that the bot is no longer monitoring any channel. """
    message = "The bot is no longer monitoring any channel."
    return message


@bot.tree.command(name="setchannel", description="Sets or clears the channel for monitoring")
async def set_channel(interaction: discord.Interaction, channel: discord.TextChannel = None, clear: bool = False):
    """ Sets the channel for the bot to monitor or clears it. """
    global CHANNEL_ID, message_counter

    with open("Discord-Bot/src/config/config.json", "r") as f:
        config = json.load(f)

    if clear or channel is None:
        CHANNEL_ID = None
        message_counter = 0
        config["channel_id"] = None  # Only modify channel_id, keeping other settings

        await interaction.response.send_message("🛑 Channel monitoring has been **stopped**.")
        print('Bot is no longer monitoring any channel.')
    else:
        CHANNEL_ID = channel.id
        message_counter = 0
        config["channel_id"] = CHANNEL_ID

        await interaction.response.send_message(f'✅ Channel set to {channel.mention}')
        print(f'Bot is now monitoring channel: {channel.name} (ID: {channel.id})')

    with open("Discord-Bot/src/config/config.json", "w") as f:
        json.dump(config, f, indent=4)


@bot.tree.command(name="ping", description="Responds with the bot's latency")
async def ping(interaction: discord.Interaction):
    """ Responds with the bot's latency, including API response time. """
    start_time = time.perf_counter()  
    await interaction.response.defer(thinking=True)  # Prevent timeout
    end_time = time.perf_counter()  
    heartbeat_latency = round(bot.latency * 1000, 2)
    api_latency = round((end_time - start_time) * 1000, 2)
    print(f"Pong! Heartbeat: {heartbeat_latency}ms | API Latency: {api_latency}ms")

    await interaction.followup.send(f"🏓 **Pong!**\n⏳ Heartbeat: `{heartbeat_latency}ms`\n📡 API Latency: `{api_latency}ms`")


@bot.tree.command(name="chat", description="Chat with the GPT-2 model")
async def chat(interaction: discord.Interaction, message: str):
    """ Responds with a message from the GPT-2 model. """
    global message_counter, CHANNEL_ID

    await interaction.response.defer(thinking=True)  

    print("Thinking...")
    llm_response = await run_gpt_model(user_input=message)
    print("Chatting")

    if not llm_response.strip():
        llm_response = "*[No response generated]*"

    await interaction.followup.send(f"{llm_response}")

    if CHANNEL_ID and interaction.channel_id == CHANNEL_ID:
        message_counter += 1
        print(f"✅ Message counter updated: {message_counter}")


@bot.tree.command(name="power", description="Reboot or shut down the bot.")
@app_commands.choices(action=[
    app_commands.Choice(name="Reboot", value="reboot"),
    app_commands.Choice(name="Shutdown", value="off")
])
async def power(interaction: discord.Interaction, action: app_commands.Choice[str]):
    """ Handles bot restart or shutdown based on the chosen action. """
    global CHANNEL_ID

    # Save the reboot channel only if restarting
    if action.value == "reboot":
        with open("Discord-Bot/src/config/config.json", "r") as f:
            config = json.load(f)
        config["reboot_channel"] = interaction.channel_id
        with open("Discord-Bot/src/config/config.json", "w") as f:
            json.dump(config, f, indent=4)

    llm_message = await run_gpt_model(user_input="Give me a message before I kill you")

    if action.value == "reboot":
        await interaction.response.send_message(f"{llm_message}\n\n🔄 Restarting bot...")
        print("🔄 Restarting bot...")
        with open("restart.flag", "w") as f:
            f.write("restart")
    else:
        await interaction.response.send_message(f"{llm_message}\n\n🛑 Shutting down bot...")
        print("🛑 Shutting down bot...")

    await bot.close()
    sys.exit(0)

@bot.tree.command(name="sync", description="Manually sync bot commands.")
async def sync(interaction: discord.Interaction):
    """ Manually sync/update bot commands. """
    await interaction.response.defer(thinking=True)  # Prevent timeout
    try:
        synced = await bot.tree.sync()  # ✅ Sync all commands
        await interaction.followup.send(f"✅ Synced {len(synced)} commands.")
        print(f"✅ Manually synced {len(synced)} commands.")
    except Exception as e:
        await interaction.followup.send(f"❌ Failed to sync commands: `{e}`")
        print(f"❌ Failed to sync commands: {e}")




@bot.event
async def on_message(message):
    """ Processes messages and extracts media context """
    global CHANNEL_ID, message_counter, recent_messages
    if message.author == bot.user:
        return

    content = message.content
    extracted_info = []

    if message.attachments:
        print(f"Message has attachments")
        for attachment in message.attachments:
            if attachment.filename.lower().endswith(("png", "jpg", "jpeg","gif")):
                img_data = requests.get(attachment.url).content
                img = Image.open(io.BytesIO(img_data))

                # OCR Extraction
                extracted_text = extract_text_from_image(img)
                if extracted_text:
                    extracted_info.append(f"📖 **OCR Text:** {extracted_text}")

                # Image Captioning
                caption = generate_image_caption(img)
                extracted_info.append(f"🖼 **Caption:** {caption}")

                # YOLOv8 Object Detection
                yolo_objects = detect_objects_with_yolo(img)
                extracted_info.append(f"🔍 **Detected Objects:** {', '.join(yolo_objects)}")

            elif attachment.filename.lower().endswith(("mp4", "mov")):
                # Video Thumbnail Extraction
                thumbnail_path = extract_video_thumbnail(attachment.url)
                if thumbnail_path:
                    extracted_info.append(f"🎬 **Extracted Thumbnail:** {thumbnail_path}")
        
                    # Run image captioning & object detection on thumbnail
                    thumbnail_img = Image.open(thumbnail_path)
                    caption = generate_image_caption(thumbnail_img)
                    yolo_objects = detect_objects_with_yolo(thumbnail_img)

                    extracted_info.append(f"🖼 **Thumbnail Caption:** {caption}")
                    extracted_info.append(f"🔍 **Thumbnail Objects:** {', '.join(yolo_objects)}")

    # Handle Tenor GIFs
    if "tenor.com" in content:
        tenor_url_match = re.search(r'https://tenor\.com/view/[^\s]+', content)
        if tenor_url_match:
            tenor_url = tenor_url_match.group(0)
            tenor_gif_url = await get_tenor_gif_url(tenor_url)
            if tenor_gif_url:
                extracted_info.append(f"🎭 **Tenor GIF:** {tenor_gif_url}")
                # Analyze the GIF
                gif_data = requests.get(tenor_gif_url).content
                gif_img = Image.open(io.BytesIO(gif_data))

                # OCR Extraction
                extracted_text = extract_text_from_image(gif_img)
                if extracted_text:
                    extracted_info.append(f"📖 **GIF OCR Text:** {extracted_text}")

                # Image Captioning
                caption = generate_image_caption(gif_img)
                extracted_info.append(f"🖼 **GIF Caption:** {caption}")

                # YOLOv8 Object Detection
                yolo_objects = detect_objects_with_yolo(gif_img)
                extracted_info.append(f"🔍 **GIF Detected Objects:** {', '.join(yolo_objects)}")
            else:
                extracted_info.append(f"🎭 **Tenor GIF:** [Failed to fetch direct URL]({embed.url})")

    full_message = f"📩 **Message:** {content}" + "\n".join(extracted_info)
    print(f'📨 Message {message_counter} in channel: {full_message}')

    if CHANNEL_ID and message.channel.id == CHANNEL_ID:
        message_counter += 1  
        recent_messages.append(full_message)  # Add the message to recent_messages
        print(f"✅ Message counter updated: {message_counter}")

        if message_counter % 3 == 0:
            print('Thinking...')
            combined_messages = "\n\n".join(recent_messages[-3:])  # Combine the last 3 messages
            llm_response = await run_gpt_model(user_input=combined_messages)
            print(f'combined_messages: {combined_messages}')
            print('Chatting')
            if not llm_response.strip():
                llm_response = "*[No response generated]*"

            await message.channel.send(f"{llm_response}")

    await bot.process_commands(message)



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
