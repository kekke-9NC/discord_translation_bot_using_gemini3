import discord
from discord.ext import commands
import asyncio

# Gemini API用ライブラリ
# Gemini API用ライブラリ
from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig

import os
import aiohttp
import random
import csv
import datetime
import json

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

# --- LM Studio設定 ---
ENABLE_COMPARISON = False  # True にすると A/B テストが有効になる
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
LM_MODELS = ["qwen/qwen3-vl-4b", "google/gemma-3-4b"]
# 推奨パラメータ設定
LM_MODEL_PARAMS = {
    "qwen/qwen3-vl-4b": {"temperature": 0.7, "top_p": 0.8},
    "google/gemma-3-4b": {"temperature": 1.0, "top_p": 0.95},
}
# LM Studioの処理を直列化するためのロック
lm_lock = asyncio.Lock()
# 比較結果のログファイル
COMPARISON_LOG_FILE = "comparison_log.csv"

# ログファイルがなければヘッダーを作成
if not os.path.exists(os.path.join(BASE_DIR, COMPARISON_LOG_FILE)):
    with open(os.path.join(BASE_DIR, COMPARISON_LOG_FILE), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "SourceChannelID", "MessageID", "ModelA", "ModelB", "Selected", "WinnerModel"])


# チャンネルIDのペア定義（日本語→英語）
CHANNEL_JA_EN_PAIRS = {
    1364278336301436936: 1430253738324660295,
}
#    638712629589704709: 1340379759259025481,
#    589466932277674041: 1340414725527441439,
#    514729798451593256: 850755778087354368,
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
        "あなたはプロの翻訳者です。指示に従って翻訳を行ってください。\n\n"
        "### 会話の文脈 (翻訳しないでください):\n"
        "以下の内容は会話の流れを理解するための参考情報です。この内容自体を翻訳したり、出力に含めたりしないでください。\n"
        f"--- CONTEXT START ---\n{context}\n--- CONTEXT END ---\n\n"
        "### 翻訳指示:\n"
        "入力された日本語の文章を、原文の意味やニュアンスを損なわず、正確に英語へ翻訳してください。\n"
        "Discord内の絵文字やメンション（例：:smile: や @username）は翻訳せず、そのまま出力してください。\n"
        "翻訳結果以外の余計な文章（「翻訳結果：」など）は一切出力しないでください。"
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

# 英語→日本語の翻訳（context を含む）
async def translate_to_japanese(text: str, context: str = "") -> str:
    system_instruction = (
        "You are a professional translator. Follow the instructions below.\n\n"
        "### Conversation Context (DO NOT TRANSLATE):\n"
        "The following content is for reference only to understand the conversation flow. Do not translate this content or include it in the output.\n"
        f"--- CONTEXT START ---\n{context}\n--- CONTEXT END ---\n\n"
        "### Translation Instructions:\n"
        "Translate the input English text into Japanese accurately while preserving its original nuance and meaning.\n"
        "Do not translate Discord emojis or mentions (e.g., :smile: or @username); output them as is.\n"
        "Output ONLY the translated text without any preamble or explanation."
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

# --- LM Studio Comparison Logic ---

async def query_lm_studio(model: str, messages: list) -> str:
    """LM StudioのOpenAI互換APIを叩く"""
    # パラメータ取得 (デフォルトは temp=0.7, top_p=0.9)
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

        await interaction.response.send_message(f"回答ありがとうございます！ (選択: {selection})", ephemeral=True)
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

    @discord.ui.button(label="どちらも同じ", style=discord.ButtonStyle.secondary)
    async def button_same(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.log_result(interaction, "Same", "Draw")

    @discord.ui.button(label="わからない", style=discord.ButtonStyle.secondary)
    async def button_unknown(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.log_result(interaction, "Unknown", "N/A")

async def run_comparison_task(message: discord.Message, context: str):
    """LM Studioモデルでの翻訳を実行し、比較投票パネルを表示する"""
    original_text = message.content
    target_language = "English" if is_japanese(original_text) else "Japanese"
    
    # プロンプト作成 (文脈と指示を明確に分離)
    if target_language == "English":
        system_instruction = (
            "あなたはプロの翻訳者です。\n"
            "### 会話の文脈 (翻訳禁止):\n"
            f"--- CONTEXT START ---\n{context}\n--- CONTEXT END ---\n\n"
            "### 翻訳指示:\n"
            "上記の文脈を理解の助けとした上で、以下の文章をプロフェッショナルな英語に翻訳してください。\n"
            "Discordの絵文字やメンションはそのままにしてください。\n"
            "翻訳結果のみを出力してください。文脈の内容を翻訳に混ぜないでください。"
        )
    else:
        system_instruction = (
            "あなたはプロの翻訳者です。\n"
            "### 会話の文脈 (翻訳禁止):\n"
            f"--- CONTEXT START ---\n{context}\n--- CONTEXT END ---\n\n"
            "### 翻訳指示:\n"
            "上記の文脈を理解の助けとした上で、以下の文章をプロフェッショナルな日本語に翻訳してください。\n"
            "Discordの絵文字やメンションはそのままにしてください。\n"
            "翻訳結果のみを出力してください。文脈の内容を翻訳に混ぜないでください。"
        )

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": original_text}
    ]

    translations = {}

    # モデル直列実行 (VRAM節約)
    async with lm_lock:
        print(f"Starting comparison for message {message.id}...")
        for model in LM_MODELS:
            print(f"Querying {model}...")
            # ここでモデル切り替えのロードが発生する（LM Studioの仕様依存）
            # 完了するまで待つ
            result = await query_lm_studio(model, messages)
            translations[model] = result
            print(f"Finished {model}.")
            # 少し待機してリソース解放の猶予を与える（念のため）
            await asyncio.sleep(1)

    # A/B ランダム化
    models = list(translations.keys())
    if len(models) < 2:
        return # エラーなどで揃わなかった場合

    random.shuffle(models)
    model_a = models[0]
    model_b = models[1]
    
    # embed作成
    embed = discord.Embed(title="翻訳モデル品質比較アンケート", description="どちらの翻訳がより自然で正確ですか？", color=0x00ff00)
    embed.add_field(name="原文", value=original_text, inline=False)
    embed.add_field(name="モデル A", value=translations[model_a], inline=False)
    embed.add_field(name="モデル B", value=translations[model_b], inline=False)
    
    view = ComparisonView(model_a, model_b, translations[model_a], translations[model_b], message.channel.id, message.id)
    
    await message.channel.send(content="翻訳品質向上のため、ご協力をお願いします！", embed=embed, view=view)

class SuggestReplyView(discord.ui.View):
    def __init__(self, source_channel_id):
        super().__init__(timeout=None)
        self.source_channel_id = source_channel_id

    @discord.ui.button(label="返信サジェスト (Gemini-3 Flash)", style=discord.ButtonStyle.success, emoji="💡")
    async def suggest_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer(ephemeral=True)

        context = get_conversation_context(self.source_channel_id)
        
        system_instruction = (
            f"以下はチャットの会話ログです:\n{context}\n"
            "あなたは、この会話に参加しているユーザーのひとりとして、自然な返信を考えてください。"
            "直近のメッセージに対する返答として適切なものを1つ提案してください。"
            "返信案のみを出力してください（「返信案:」などの接頭辞は不要）。"
        )

        try:
            response = await async_client.models.generate_content(
                model="gemini-3-flash-preview",
                contents="返信案を考えて。",
                config=GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.7, 
                    candidate_count=1,
                    thinking_config=ThinkingConfig(include_thoughts=False)
                )
            )
            suggestion = response.text.strip()
        except Exception as e:
            suggestion = f"生成エラー: {e}"

        await interaction.followup.send(content=f"💡 **返信サジェスト (Gemini-3 Flash)**:\n\n{suggestion}", ephemeral=True)

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# 提案メッセージ管理: channel_id -> message object
suggestion_messages = {}

@bot.event
async def on_ready():
    print(f"Bot is ready. Logged in as {bot.user}")
    
    # 全監視チャンネルの履歴をキャッシュ
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
            # historyは新しい順に取れるので、古い順（会話順）に直す
            conversation_memory[channel_id] = history[::-1]
            print(f"Cached {len(history)} messages for channel {channel.name} ({channel_id})")
        except Exception as e:
            print(f"Failed to cache history for channel {channel_id}: {e}")
    print("Conversation history cached.")

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
    
    # 転送メッセージ自体にはViewを付けない
    forwarded = await target_channel.send(content=content_to_send, reference=ref, mention_author=False, allowed_mentions=allowed_mentions)

    forward_map[(source_channel_id, message.id)] = forwarded
    forward_map[(target_channel.id, forwarded.id)] = message

    # サジェストボタンを「元のチャンネル」に送信（自分だけに送信、という概念はBotではEphemeral以外難しいが、
    # ユーザー要望は「送信した元のチャンネルに送られるように」かつ「自分にだけ（前の文脈から見ると）」
    # しかしBotから通常メッセージとして送ると全員に見える。EphemeralはInteraction応答でしか使えない。
    # ここでは要望通り「元のチャンネルに送る」を優先し、普通のメッセージとして送る。
    # 30秒で消えるので邪魔にはなりにくい。
    
    if message.content:
        # 既存のサジェストメッセージがあれば消す
        if source_channel_id in suggestion_messages:
            try:
                await suggestion_messages[source_channel_id].delete()
            except:
                pass
            del suggestion_messages[source_channel_id]

        view = SuggestReplyView(source_channel_id)
        try:
            suggestion_msg = await message.channel.send("💬 返信サジェスト (30秒で消えます)", view=view, delete_after=30)
            suggestion_messages[source_channel_id] = suggestion_msg
        except Exception as e:
            print(f"Failed to send suggestion message: {e}")

    # LM Studio比較タスクをバックグラウンドで開始
    if ENABLE_COMPARISON and message.content:
         asyncio.create_task(run_comparison_task(message, context if 'context' in locals() else get_conversation_context(source_channel_id)))

    return forwarded

@bot.event
async def on_message(message: discord.Message):
    # 会話が進んだら古いサジェストを消す
    if message.channel.id in suggestion_messages:
        # 自分が送ったサジェストメッセージ自身でなければ消す（ユーザーの発言等）
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
