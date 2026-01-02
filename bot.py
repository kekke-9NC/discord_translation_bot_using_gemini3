import discord
from discord.ext import commands
import asyncio
import os
import aiohttp
import random
import csv
import datetime
import json

# Gemini API Library
from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig

# --- Configuration ---
# Get the directory of the executable file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def read_credential(filename):
    try:
        with open(os.path.join(BASE_DIR, filename), 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        return ""

TOKEN = read_credential("discord_token.txt")
GEMINI_API_KEY = read_credential("gemini_api_key.txt")

# --- LM Studio Settings ---
ENABLE_COMPARISON = False  # Set to True to enable A/B testing
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
LM_MODELS = ["qwen/qwen3-vl-4b", "google/gemma-3-4b"]

# Recommended Parameter Settings
LM_MODEL_PARAMS = {
    "qwen/qwen3-vl-4b": {"temperature": 0.7, "top_p": 0.8},
    "google/gemma-3-4b": {"temperature": 1.0, "top_p": 0.95},
}

# Lock for serializing LM Studio processing (VRAM conservation)
lm_lock = asyncio.Lock()

# Log file for comparison results
COMPARISON_LOG_FILE = "comparison_log.csv"

# Initialize log file with header if it doesn't exist
if not os.path.exists(os.path.join(BASE_DIR, COMPARISON_LOG_FILE)):
    with open(os.path.join(BASE_DIR, COMPARISON_LOG_FILE), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "SourceChannelID", "MessageID", "ModelA", "ModelB", "Selected", "WinnerModel"])

# Channel pair definitions (Japanese <-> English)
CHANNEL_JA_EN_PAIRS = {
    1364278336301436936: 1430253738324660295,
}

# Reverse mapping for English to Japanese
CHANNEL_EN_JA = {v: k for k, v in CHANNEL_JA_EN_PAIRS.items()}

# Mapping (channel_id, message_id) -> target Discord message object
forward_map = {}

# Conversation history: keeps the last 10 messages for each target channel
conversation_memory = {}

gemini_client = genai.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1alpha'})
async_client = gemini_client.aio

def is_japanese(text: str) -> bool:
    """Checks if the text contains Japanese characters."""
    for ch in text:
        if ('\u3040' <= ch <= '\u30ff') or ('\u4e00' <= ch <= '\u9faf'):
            return True
    return False

def get_conversation_context(channel_id: int) -> str:
    """Returns the conversation history for a specific channel."""
    messages = conversation_memory.get(channel_id, [])
    return "\n".join(messages)

async def translate_to_english(text: str, context: str = "") -> str:
    """Translates Japanese text to English using Gemini."""
    system_instruction = (
        "You are a professional translator. Follow the instructions below.\n\n"
        "### Conversation Context (DO NOT TRANSLATE):\n"
        "The following content is for reference only. Do not translate this content or include it in the output.\n"
        f"--- CONTEXT START ---\n{context}\n--- CONTEXT END ---\n\n"
        "### Translation Instructions:\n"
        "Translate the input Japanese text into English accurately while preserving its original nuance and meaning.\n"
        "Do not translate Discord emojis or mentions (e.g., :smile: or @username); output them as is.\n"
        "Output ONLY the translated text without any preamble."
    )
    response = await async_client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=text,
        config=GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0,
            candidate_count=1,
            thinking_config=ThinkingConfig(include_thoughts=False)
        )
    )
    return response.text.strip()

async def translate_to_japanese(text: str, context: str = "") -> str:
    """Translates English text to Japanese using Gemini."""
    system_instruction = (
        "You are a professional translator. Follow the instructions below.\n\n"
        "### Conversation Context (DO NOT TRANSLATE):\n"
        "The following content is for reference only. Do not translate this content or include it in the output.\n"
        f"--- CONTEXT START ---\n{context}\n--- CONTEXT END ---\n\n"
        "### Translation Instructions:\n"
        "Translate the input English text into Japanese accurately while preserving its original nuance and meaning.\n"
        "Do not translate Discord emojis or mentions (e.g., :smile: or @username); output them as is.\n"
        "Output ONLY the translated text without any preamble."
    )
    response = await async_client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=text,
        config=GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0,
            candidate_count=1,
            thinking_config=ThinkingConfig(include_thoughts=False)
        )
    )
    return response.text.strip()

async def safe_translate_to_english(text: str, context: str, max_retries: int = 2) -> str:
    """Wrapper for translate_to_english with retries and language verification."""
    last_result = ""
    for attempt in range(max_retries + 1):
        try:
            result = await translate_to_english(text, context=context)
        except Exception:
            result = ""
        if result and not is_japanese(result):
            return result
        last_result = result
        await asyncio.sleep(1)
    return last_result if last_result else "Translation error."

async def safe_translate_to_japanese(text: str, context: str, max_retries: int = 2) -> str:
    """Wrapper for translate_to_japanese with retries and language verification."""
    last_result = ""
    for attempt in range(max_retries + 1):
        try:
            result = await translate_to_japanese(text, context=context)
        except Exception:
            result = ""
        if result and is_japanese(result):
            return result
        last_result = result
        await asyncio.sleep(1)
    return last_result if last_result else "Translation error (JP)."

# --- LM Studio Comparison Logic ---

async def query_lm_studio(model: str, messages: list) -> str:
    """Calls the OpenAI-compatible API of LM Studio."""
    params = LM_MODEL_PARAMS.get(model, {"temperature": 0.7, "top_p": 0.9})
    
    async with aiohttp.ClientSession() as session:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": params["temperature"],
            "top_p": params["top_p"],
        }
        try:
            async with session.post(LM_STUDIO_URL, json=payload, timeout=120) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data['choices'][0]['message']['content'].strip()
                else:
                    return f"Error: {resp.status}"
        except Exception as e:
            return f"Error: {e}"

class ComparisonView(discord.ui.View):
    """View class for A/B testing voting panel."""
    def __init__(self, model_a_name, model_b_name, model_a_text, model_b_text, source_channel_id, message_id):
        super().__init__(timeout=None)
        self.model_a_name = model_a_name
        self.model_b_name = model_b_name
        self.model_a_text = model_a_text
        self.model_b_text = model_b_text
        self.source_channel_id = source_channel_id
        self.message_id = message_id

    async def log_result(self, interaction: discord.Interaction, selection: str, winner_model: str):
        filepath = os.path.join(BASE_DIR, COMPARISON_LOG_FILE)
        timestamp = datetime.datetime.now().isoformat()
        try:
            with open(filepath, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, self.source_channel_id, self.message_id, self.model_a_name, self.model_b_name, selection, winner_model])
        except Exception as e:
            print(f"Logging error: {e}")

        await interaction.response.send_message(f"Thanks for voting! (Selected: {selection})", ephemeral=True)
        try:
            await interaction.message.delete()
        except Exception as e:
            print(f"Message delete error: {e}")

    @discord.ui.button(label="A", style=discord.ButtonStyle.primary)
    async def button_a(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.log_result(interaction, "A", self.model_a_name)

    @discord.ui.button(label="B", style=discord.ButtonStyle.primary)
    async def button_b(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.log_result(interaction, "B", self.model_b_name)

    @discord.ui.button(label="Both Same", style=discord.ButtonStyle.secondary)
    async def button_same(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.log_result(interaction, "Same", "Draw")

    @discord.ui.button(label="N/A", style=discord.ButtonStyle.secondary)
    async def button_unknown(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.log_result(interaction, "Unknown", "N/A")

async def run_comparison_task(message: discord.Message, context: str):
    """Executes translations using LM Studio models and displays the voting panel."""
    original_text = message.content
    target_language = "English" if is_japanese(original_text) else "Japanese"
    
    if target_language == "English":
        system_instruction = (
            "You are a professional translator.\n"
            "### Conversation Context (DO NOT TRANSLATE):\n"
            f"--- CONTEXT START ---\n{context}\n--- CONTEXT END ---\n\n"
            "### Translation Instructions:\n"
            "Translate the following text into professional English using the above context as a guide.\n"
            "Keep Discord emojis and mentions intact.\n"
            "Output ONLY the translation."
        )
    else:
        system_instruction = (
            "You are a professional translator.\n"
            "### Conversation Context (DO NOT TRANSLATE):\n"
            f"--- CONTEXT START ---\n{context}\n--- CONTEXT END ---\n\n"
            "### Translation Instructions:\n"
            "Translate the following text into professional Japanese using the above context as a guide.\n"
            "Keep Discord emojis and mentions intact.\n"
            "Output ONLY the translation."
        )

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": original_text}
    ]

    translations = {}

    # Sequential execution to save VRAM
    async with lm_lock:
        print(f"Starting comparison for message {message.id}...")
        for model in LM_MODELS:
            print(f"Querying {model}...")
            result = await query_lm_studio(model, messages)
            translations[model] = result
            print(f"Finished {model}.")
            await asyncio.sleep(1)

    models = list(translations.keys())
    if len(models) < 2:
        return

    random.shuffle(models)
    model_a = models[0]
    model_b = models[1]
    
    embed = discord.Embed(title="Translation Quality Comparison", description="Which translation is more natural and accurate?", color=0x00ff00)
    embed.add_field(name="Original", value=original_text, inline=False)
    embed.add_field(name="Model A", value=translations[model_a], inline=False)
    embed.add_field(name="Model B", value=translations[model_b], inline=False)
    
    view = ComparisonView(model_a, model_b, translations[model_a], translations[model_b], message.channel.id, message.id)
    await message.channel.send(content="Please help us improve translation quality!", embed=embed, view=view)

class SuggestReplyView(discord.ui.View):
    """View class for reply suggestion button."""
    def __init__(self, source_channel_id):
        super().__init__(timeout=None)
        self.source_channel_id = source_channel_id

    @discord.ui.button(label="Suggest Reply (Gemini-3 Flash)", style=discord.ButtonStyle.success, emoji="💡")
    async def suggest_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer(ephemeral=True)

        context = get_conversation_context(self.source_channel_id)
        
        system_instruction = (
            f"Conversation Log:\n{context}\n"
            "Act as a participant in this conversation and provide one natural reply suggestion.\n"
            "Output ONLY the reply (no prefixes)."
        )

        try:
            response = await async_client.models.generate_content(
                model="gemini-3-flash-preview",
                contents="Generate a reply suggestion.",
                config=GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.7, 
                    candidate_count=1,
                    thinking_config=ThinkingConfig(include_thoughts=False)
                )
            )
            suggestion = response.text.strip()
        except Exception as e:
            suggestion = f"Generation error: {e}"

        await interaction.followup.send(content=f"💡 **Reply Suggestion (Gemini-3 Flash)**:\n\n{suggestion}", ephemeral=True)

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Map for suggestion messages: channel_id -> message object
suggestion_messages = {}

@bot.event
async def on_ready():
    print(f"Bot is ready. Logged in as {bot.user}")
    
    # Cache conversation history for all monitored channels
    target_channels = set(CHANNEL_JA_EN_PAIRS.keys()) | set(CHANNEL_EN_JA.keys())
    print("Caching conversation history...")
    for channel_id in target_channels:
        channel = bot.get_channel(channel_id)
        if not channel:
            continue
        try:
            history = []
            async for msg in channel.history(limit=10):
                if not msg.author.bot:
                     history.append(f"{msg.author.display_name}: {msg.content}")
            # Cache in chronological order
            conversation_memory[channel_id] = history[::-1]
            print(f"Cached {len(history)} messages for channel {channel.name} ({channel_id})")
        except Exception as e:
            print(f"Failed to cache history for channel {channel_id}: {e}")
    print("Conversation history cached.")

def build_forward_content(message: discord.Message, translated_text: str) -> str:
    """Builds the content for the forwarded message including image links."""
    header = f"__**{message.author.mention}**__\n"
    content = header + translated_text if translated_text else header
    image_links = []
    for attachment in message.attachments:
        if attachment.content_type and attachment.content_type.startswith("image"):
            image_links.append(attachment.url)
    if image_links:
        content += "\n" + "\n".join(image_links)
    return content

async def forward_message(message: discord.Message) -> discord.Message:
    """Processes and forwards a message to the linked channel."""
    source_channel_id = message.channel.id
    # Update memory
    conversation_memory.setdefault(source_channel_id, [])
    conversation_memory[source_channel_id].append(f"{message.author.display_name}: {message.content}")
    conversation_memory[source_channel_id] = conversation_memory[source_channel_id][-10:]
    
    # Self-mapping (internal translation within the same channel)
    if source_channel_id in CHANNEL_JA_EN_PAIRS and CHANNEL_JA_EN_PAIRS[source_channel_id] == source_channel_id:
        target_channel = message.channel
        if message.content:
            context = get_conversation_context(source_channel_id)
            if is_japanese(message.content):
                try:
                    translated = await safe_translate_to_english(message.content, context=context)
                except Exception:
                    translated = "Translation error."
            else:
                try:
                    translated = await safe_translate_to_japanese(message.content, context=context)
                except Exception:
                    translated = "Translation error (JP)."
        else:
            translated = ""
    # Standard forward (Japanese -> English)
    elif source_channel_id in CHANNEL_JA_EN_PAIRS:
        target_channel = bot.get_channel(CHANNEL_JA_EN_PAIRS[source_channel_id])
        if message.content:
            if is_japanese(message.content):
                context = get_conversation_context(source_channel_id)
                try:
                    translated = await safe_translate_to_english(message.content, context=context)
                except Exception:
                    translated = "Translation error."
            else:
                translated = message.content
        else:
            translated = ""
    # English -> Japanese forward
    elif source_channel_id in CHANNEL_EN_JA:
        target_channel = bot.get_channel(CHANNEL_EN_JA[source_channel_id])
        if message.content:
            if not is_japanese(message.content):
                context = get_conversation_context(source_channel_id)
                try:
                    translated = await safe_translate_to_japanese(message.content, context=context)
                except Exception:
                    translated = "Translation error (JP)."
            else:
                translated = message.content
        else:
            translated = ""
    else:
        return None

    ref = None
    if message.reference and message.reference.message_id:
        key = (source_channel_id, message.reference.message_id)
        if key in forward_map:
            ref = forward_map[key]

    if ref is not None and ref.channel.id != target_channel.id:
        ref = None

    content_to_send = build_forward_content(message, translated)
    
    allowed_mentions = discord.AllowedMentions(
        users=False,
        replied_user=False
    )
    
    forwarded = await target_channel.send(content=content_to_send, reference=ref, mention_author=False, allowed_mentions=allowed_mentions)

    forward_map[(source_channel_id, message.id)] = forwarded
    forward_map[(target_channel.id, forwarded.id)] = message

    # Send suggestion button (deletes after 30s)
    if message.content:
        if source_channel_id in suggestion_messages:
            try:
                await suggestion_messages[source_channel_id].delete()
            except:
                pass
            del suggestion_messages[source_channel_id]

        view = SuggestReplyView(source_channel_id)
        try:
            suggestion_msg = await message.channel.send("💬 Reply suggestion (Deletes in 30s)", view=view, delete_after=30)
            suggestion_messages[source_channel_id] = suggestion_msg
        except Exception as e:
            print(f"Failed to send suggestion message: {e}")

    # Start A/B testing task if enabled
    if ENABLE_COMPARISON and message.content:
         asyncio.create_task(run_comparison_task(message, context if 'context' in locals() else get_conversation_context(source_channel_id)))

    return forwarded

@bot.event
async def on_message(message: discord.Message):
    # Remove old suggestion message on new activity
    if message.channel.id in suggestion_messages:
        msg = suggestion_messages[message.channel.id]
        if msg and msg.id != message.id: 
            try:
                await msg.delete()
            except:
                pass
            del suggestion_messages[message.channel.id]

    if message.channel.id in CHANNEL_JA_EN_PAIRS or message.channel.id in CHANNEL_EN_JA:
        forward_map[(message.channel.id, message.id)] = message
        conversation_memory.setdefault(message.channel.id, [])
        conversation_memory[message.channel.id].append(f"{message.author.display_name}: {message.content}")
        conversation_memory[message.channel.id] = conversation_memory[message.channel.id][-10:]
    
    if message.author.bot:
        return
        
    if message.channel.id in CHANNEL_JA_EN_PAIRS or message.channel.id in CHANNEL_EN_JA:
        await forward_message(message)
    else:
        await bot.process_commands(message)

@bot.event
async def on_message_edit(before: discord.Message, after: discord.Message):
    if after.author.bot:
        return
    source_channel_id = after.channel.id
    if source_channel_id not in CHANNEL_JA_EN_PAIRS and source_channel_id not in CHANNEL_EN_JA:
        return
    
    conversation_memory.setdefault(source_channel_id, [])
    conversation_memory[source_channel_id].append(f"{after.author.display_name}: {after.content}")
    conversation_memory[source_channel_id] = conversation_memory[source_channel_id][-10:]
    
    key = (source_channel_id, after.id)
    if key not in forward_map:
        return
    forwarded_msg = forward_map[key]

    # Process edits for self-mapping or paired channels
    if source_channel_id in CHANNEL_JA_EN_PAIRS and CHANNEL_JA_EN_PAIRS[source_channel_id] == source_channel_id:
        if after.content:
            context = get_conversation_context(source_channel_id)
            if is_japanese(after.content):
                try:
                    translated = await safe_translate_to_english(after.content, context=context)
                except Exception:
                    translated = "Translation error."
            else:
                try:
                    translated = await safe_translate_to_japanese(after.content, context=context)
                except Exception:
                    translated = "Translation error (JP)."
        else:
            translated = ""
    elif source_channel_id in CHANNEL_JA_EN_PAIRS:
        if after.content:
            if is_japanese(after.content):
                context = get_conversation_context(source_channel_id)
                try:
                    translated = await safe_translate_to_english(after.content, context=context)
                except Exception:
                    translated = "Translation error."
            else:
                translated = after.content
        else:
            translated = ""
    else:
        if after.content:
            if not is_japanese(after.content):
                context = get_conversation_context(source_channel_id)
                try:
                    translated = await safe_translate_to_japanese(after.content, context=context)
                except Exception:
                    translated = "Translation error (JP)."
            else:
                translated = after.content
        else:
            translated = ""
            
    new_content = build_forward_content(after, translated)
    try:
        await forwarded_msg.edit(content=new_content)
    except Exception as e:
        print("Edit error:", e)

@bot.event
async def on_message_delete(message: discord.Message):
    if message.author.bot:
        return
    source_channel_id = message.channel.id
    if source_channel_id not in CHANNEL_JA_EN_PAIRS and source_channel_id not in CHANNEL_EN_JA:
        return
    
    key = (source_channel_id, message.id)
    if key in forward_map:
        forwarded_msg = forward_map.pop(key)
        key_target = (forwarded_msg.channel.id, forwarded_msg.id)
        if key_target in forward_map:
            forward_map.pop(key_target)
        try:
            await forwarded_msg.delete()
        except Exception as e:
            print("Delete error:", e)

bot.run(TOKEN)
