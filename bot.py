import discord
from discord.ext import commands
import asyncio

# Gemini API用ライブラリ
from google import genai
from google.genai.types import GenerateContentConfig

import os

# --- 設定 ---
# 実行ファイルのディレクトリを取得
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

# チャンネルIDのペア定義（日本語→英語）
CHANNEL_JA_EN_PAIRS = {
    638712629589704709: 1340379759259025481,
    589466932277674041: 1340414725527441439,
    514729798451593256: 850755778087354368,
}

# 英語チャンネルから逆方向へのマッピング（英語→日本語）
CHANNEL_EN_JA = {v: k for k, v in CHANNEL_JA_EN_PAIRS.items()}

# (channel_id, message_id) -> 転送先のDiscordメッセージオブジェクト
forward_map = {}

# 会話メモリ：対象チャンネルごとに最新10件の会話を保持する
conversation_memory = {}

gemini_client = genai.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1alpha'})
async_client = gemini_client.aio

def is_japanese(text: str) -> bool:
    for ch in text:
        if ('\u3040' <= ch <= '\u30ff') or ('\u4e00' <= ch <= '\u9faf'):
            return True
    return False

def get_conversation_context(channel_id: int) -> str:
    messages = conversation_memory.get(channel_id, [])
    return "\n".join(messages)

# 日本語→英語の翻訳（context を含む）
async def translate_to_english(text: str, context: str = "") -> str:
    system_instruction = (
        f"以下の会話の文脈を考慮してください:\n{context}\n"
        "あなたはプロの翻訳者です。以下の日本語の文章を、原文の意味やニュアンスを損なわず、正確に英語へ翻訳してください。"
        "ただし、Discord内の絵文字やメンション（例：:smile: や @username）は翻訳せず、そのまま出力してください。"
        "翻訳結果以外の余計な文章は一切出力しないでください。"
    )
    response = await async_client.models.generate_content(
        model="gemini-3.0-flash",
        contents=text,
        config=GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0,
            candidate_count=1,
        )
    )
    return response.text.strip()

# 英語→日本語の翻訳（context を含む）
async def translate_to_japanese(text: str, context: str = "") -> str:
    system_instruction = (
        f"Consider the following conversation context:\n{context}\n"
        "You are a professional translator. Please translate the following English text into Japanese accurately while preserving its original nuance and meaning."
        "However, do not translate any Discord emojis or mentions (e.g., :smile: or @username); output them as is."
        "Do not include any additional text such as 'Below is the translation'—only output the translation."
    )
    response = await async_client.models.generate_content(
        model="gemini-3.0-flash",
        contents=text,
        config=GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0,
            candidate_count=1,
        )
    )
    return response.text.strip()

# 安全策として、翻訳結果が期待する言語でない場合に再試行するラッパー関数（context を渡す）
async def safe_translate_to_english(text: str, context: str, max_retries: int = 2) -> str:
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
    return last_result if last_result else "翻訳エラー。"

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"Bot is ready. Logged in as {bot.user}")

def build_forward_content(message: discord.Message, translated_text: str) -> str:
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
    source_channel_id = message.channel.id
    # 会話メモリの更新
    conversation_memory.setdefault(source_channel_id, [])
    conversation_memory[source_channel_id].append(f"{message.author.display_name}: {message.content}")
    conversation_memory[source_channel_id] = conversation_memory[source_channel_id][-10:]
    
    # 自己マッピングの場合（同じチャンネル内で完結）
    if source_channel_id in CHANNEL_JA_EN_PAIRS and CHANNEL_JA_EN_PAIRS[source_channel_id] == source_channel_id:
        target_channel = message.channel  # 同じチャンネル内で転送
        if message.content:
            context = get_conversation_context(source_channel_id)
            # 日本語の場合→英語、英語の場合→日本語に翻訳
            if is_japanese(message.content):
                try:
                    translated = await safe_translate_to_english(message.content, context=context)
                except Exception:
                    translated = "Translation error."
            else:
                try:
                    translated = await safe_translate_to_japanese(message.content, context=context)
                except Exception:
                    translated = "翻訳エラー。"
        else:
            translated = ""
    # 通常の日本語→英語転送（異なるチャンネル）
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
    # 英語→日本語転送
    elif source_channel_id in CHANNEL_EN_JA:
        target_channel = bot.get_channel(CHANNEL_EN_JA[source_channel_id])
        if message.content:
            if not is_japanese(message.content):
                context = get_conversation_context(source_channel_id)
                try:
                    translated = await safe_translate_to_japanese(message.content, context=context)
                except Exception:
                    translated = "翻訳エラー。"
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
    if ref and ref.channel.id != target_channel.id:
        ref = None

    allowed_mentions = discord.AllowedMentions(
        users=False,
        replied_user=False
    )
    forwarded = await target_channel.send(content=content_to_send, reference=ref, mention_author=False, allowed_mentions=allowed_mentions)

    forward_map[(source_channel_id, message.id)] = forwarded
    forward_map[(target_channel.id, forwarded.id)] = message

    return forwarded

@bot.event
async def on_message(message: discord.Message):
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
    
    # 自己マッピングの場合
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
                    translated = "翻訳エラー。"
        else:
            translated = ""
    # 通常の日本語→英語編集
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
    # 英語→日本語編集
    else:
        if after.content:
            if not is_japanese(after.content):
                context = get_conversation_context(source_channel_id)
                try:
                    translated = await safe_translate_to_japanese(after.content, context=context)
                except Exception:
                    translated = "翻訳エラー。"
            else:
                translated = after.content
        else:
            translated = ""
    new_content = build_forward_content(after, translated)
    allowed_mentions = discord.AllowedMentions(
        users=False,
        replied_user=False
    )
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
