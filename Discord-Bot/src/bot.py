import discord
import asyncio
import json
from discord import app_commands
from discord.ext import commands

# Load configuration
with open('C:/Users/zak/Downloads/LLM/Discord-Bot/src/config/config.json') as config_file:
    config = json.load(config_file)

TOKEN = config['bot_token']

# Initialize bot with intents
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
bot = commands.Bot(command_prefix='/', intents=intents)

# Global variables
CHANNEL_ID = None
message_counter = 0


@bot.event
async def on_ready():
    """ Called when the bot is ready. Syncs commands. """
    print(f'Logged in as {bot.user} (ID: {bot.user.id})')
    print('------')

    await bot.wait_until_ready()
    try:
        synced = await bot.tree.sync()
        print(f"✅ Synced {len(synced)} slash commands.")
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

    # Save the channel ID
    with open("channel_config.json", "w") as f:
        json.dump({"channel_id": CHANNEL_ID}, f)

    await interaction.response.send_message(f'✅ Channel set to {channel.mention}')
    print(f'Bot is now monitoring channel: {channel.name} (ID: {channel.id})')


@bot.tree.command(name="ping", description="Responds with the bot's latency")
async def ping(interaction: discord.Interaction):
    """ Responds with the bot's latency. """
    latency = round(bot.latency * 1000, 2)
    print(f"Pong! Latency: {latency}ms")
    await interaction.response.send_message(f"Pong! Latency: {latency}ms")


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


@bot.tree.command(name="chat", description="Chat with the GPT-2 model")
async def chat(interaction: discord.Interaction, message: str):
    """ Responds with a message from the GPT-2 model. """
    global message_counter, CHANNEL_ID

    await interaction.response.defer(thinking=True)

    print("Thinking...")
    llm_response = f"Simulated GPT Response: {message}" 
    print("Chatting")

    if not llm_response.strip():
        llm_response = "*[No response generated]*"

    await interaction.followup.send(f"{llm_response}")

    if CHANNEL_ID and interaction.channel_id == CHANNEL_ID:
        message_counter += 1
        print(f"✅ Message counter updated: {message_counter}")


@bot.event
async def on_message(message):
    """ Processes messages and responds if needed. """
    global CHANNEL_ID, message_counter

    if message.author == bot.user:
        return

    content = message.content
    print(f'📨 Message {message_counter} in channel: {content}')

    if CHANNEL_ID and message.channel.id == CHANNEL_ID:
        message_counter += 1
        print(f"✅ Message counter updated: {message_counter}")

    await bot.process_commands(message)  


# Run bot
bot.run(TOKEN)
