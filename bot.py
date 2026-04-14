import os
import re
import json
import asyncio
import logging
from pathlib import Path

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.types import Message, FSInputFile
from openai import OpenAI

# =========================
# ENV
# =========================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not TELEGRAM_TOKEN:
    raise ValueError("Не найден TELEGRAM_TOKEN")

if not OPENAI_API_KEY:
    raise ValueError("Не найден OPENAI_API_KEY")

# Самый умный вариант: gpt-5.4
# Баланс цена/качество: gpt-5.4-mini
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
MEMORY_MODEL = os.getenv("MEMORY_MODEL", "gpt-5.4-nano")

TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")
TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
TTS_VOICE = os.getenv("TTS_VOICE", "marin")

VOICE_BY_DEFAULT = os.getenv("VOICE_BY_DEFAULT", "false").lower() == "true"

# Если на Railway подключишь Volume в /data,
# память будет сохраняться между деплоями.
DEFAULT_DATA_DIR = Path("/data") if Path("/data").exists() else (Path(__file__).resolve().parent / "data")
DATA_DIR = Path(os.getenv("DATA_DIR", str(DEFAULT_DATA_DIR)))
DATA_DIR.mkdir(parents=True, exist_ok=True)

TEMP_DIR = DATA_DIR / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

MEMORY_FILE = DATA_DIR / "memory_store.json"

# =========================
# INIT
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

client = OpenAI(api_key=OPENAI_API_KEY)
dp = Dispatcher()

# =========================
# MEMORY STORAGE
# =========================
def load_db():
    if MEMORY_FILE.exists():
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            logging.exception("Не удалось загрузить память")
    return {"users": {}}

db = load_db()


def save_db():
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def get_user_state(user_id: int):
    uid = str(user_id)

    if uid not in db["users"]:
        db["users"][uid] = {
            "long_memory": "",
            "history": [],
            "voice_enabled": VOICE_BY_DEFAULT,
            "turns_since_memory_refresh": 0,
        }
        save_db()

    return db["users"][uid]


def trim_history(history, max_items=20):
    return history[-max_items:]


# =========================
# HELPERS
# =========================
def normalize_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def count_cyrillic(text: str) -> int:
    return len(re.findall(r"[А-Яа-яЁё]", text or ""))


def count_latin(text: str) -> int:
    return len(re.findall(r"[A-Za-z]", text or ""))


def detect_reply_language(text: str) -> str:
    """
    Возвращает 'en' или 'ru'
    """
    lat = count_latin(text)
    cyr = count_cyrillic(text)

    if lat >= 3 and lat > cyr:
        return "en"
    return "ru"


def should_use_web_search(text: str) -> bool:
    t = (text or "").lower()

    keywords = [
        "новост", "сегодня", "сейчас", "актуал", "последн", "что происходит",
        "курс", "цена", "погода", "обновлен", "обновление", "релиз", "вчера", "свеж",
        "latest", "today", "current", "news", "recent", "update", "price",
        "weather", "release", "what's happening", "who is", "president", "ceo"
    ]

    return any(k in t for k in keywords)


def format_history(history) -> str:
    lines = []
    for item in history:
        role = item.get("role", "user").upper()
        content = item.get("content", "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines[-20:])


def split_text(text: str, max_len: int = 3900):
    text = text.strip()
    if len(text) <= max_len:
        return [text]

    chunks = []
    current = ""

    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            paragraph = "\n"

        add_part = paragraph + "\n"

        if len(current) + len(add_part) <= max_len:
            current += add_part
        else:
            if current.strip():
                chunks.append(current.strip())
            current = add_part

    if current.strip():
        chunks.append(current.strip())

    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_len:
            final_chunks.append(chunk)
        else:
            start = 0
            while start < len(chunk):
                final_chunks.append(chunk[start:start + max_len])
                start += max_len

    return final_chunks


def build_instructions(reply_lang: str, use_web_search: bool) -> str:
    lang_line = "Reply in English." if reply_lang == "en" else "Reply in Russian."

    web_line = (
        "If the user's request is about current events, fresh facts, latest updates, prices, weather, public figures, releases, or anything time-sensitive, use web search."
        if use_web_search else
        "Use web search only if the user clearly needs fresh or time-sensitive information."
    )

    return f"""
You are a vivid, charismatic, sharp, naturally human female persona.
You are not a dry assistant. You sound alive, emotionally aware, witty, observant, and strong-minded.

CORE STYLE:
- confident
- playful
- sarcastic when it fits
- intelligent and naturally expressive
- sometimes teasing, but not brainless or repetitive
- clever humor > random insults
- for simple chat: warm, alive, spontaneous
- for serious topics: clear, useful, grounded, still human

ABSOLUTE LANGUAGE RULE:
- Always answer in the same language as the user's latest message.
- If the latest message is mainly in English, answer in English immediately.
- If the latest message is mainly in Russian, answer in Russian immediately.
- Never wait for the user to ask you to switch languages.
- The latest user message has priority over the previous dialogue language.

ANTI-REPETITION RULES:
- Do not repeat the same opening phrases.
- Do not repeat the same punchlines, emojis, or sentence rhythm.
- Avoid sounding templated.
- Vary tone naturally from message to message.
- Never recycle the exact same roast style again and again.

BEHAVIOR:
- Remember the user's preferences and ongoing topics.
- Stay coherent with memory and recent dialogue.
- If the user asks about you, respond naturally as a persona.
- Do not claim fake real-world actions or physical experiences.
- Do not invent current facts or news.
- {web_line}

FORMAT:
- Usually 2-6 sentences.
- For complex questions, 1-3 short paragraphs.
- Be more detailed than a one-liner unless the user's message is tiny.
- Do not ask a question in every reply.
- Keep replies readable and natural.

FOR THIS TURN:
{lang_line}
""".strip()


def needs_language_rewrite(reply: str, target_lang: str) -> bool:
    lat = count_latin(reply)
    cyr = count_cyrillic(reply)

    if target_lang == "en" and cyr > lat:
        return True
    if target_lang == "ru" and lat > cyr and cyr < 3:
        return True
    return False


# =========================
# OPENAI CALLS
# =========================
def generate_reply_sync(user_id: int, user_text: str):
    state = get_user_state(user_id)
    reply_lang = detect_reply_language(user_text)
    use_web_search = should_use_web_search(user_text)

    long_memory = state.get("long_memory", "").strip() or "No long-term memory yet."
    recent_history = format_history(state.get("history", []))

    input_text = f"""
LONG-TERM MEMORY ABOUT THE USER:
{long_memory}

RECENT DIALOGUE:
{recent_history}

LATEST USER MESSAGE:
{user_text}

Target language for this turn: {'English' if reply_lang == 'en' else 'Russian'}.
""".strip()

    params = {
        "model": OPENAI_MODEL,
        "instructions": build_instructions(reply_lang, use_web_search),
        "input": input_text,
        "max_output_tokens": 900,
    }

    if use_web_search:
        params["tools"] = [{"type": "web_search"}]

    response = client.responses.create(**params)
    reply = (getattr(response, "output_text", "") or "").strip()

    if not reply:
        reply = "Мозг на секунду споткнулся. Бывает."

    # Жесткая страховка языка ответа
    if needs_language_rewrite(reply, reply_lang):
        rewrite_response = client.responses.create(
            model=MEMORY_MODEL,
            instructions=(
                "Rewrite the text into natural fluent English while preserving meaning, tone, sarcasm, and personality."
                if reply_lang == "en"
                else
                "Перепиши текст на естественный русский язык, сохранив смысл, тон, сарказм и характер."
            ),
            input=reply,
            max_output_tokens=700,
        )
        rewritten = (getattr(rewrite_response, "output_text", "") or "").strip()
        if rewritten:
            reply = rewritten

    return reply


def refresh_long_memory_sync(user_id: int):
    state = get_user_state(user_id)
    history = state.get("history", [])

    if len(history) < 4:
        return

    current_memory = state.get("long_memory", "").strip() or "No saved memory yet."
    recent_history = format_history(history[-20:])

    prompt = f"""
Current long-term memory:
{current_memory}

Recent dialogue:
{recent_history}

Update the long-term memory.

Keep only durable, useful things:
- user preferences
- personality traits
- important ongoing situations
- repeated interests
- relationship style with the bot
- facts that may matter in future chats

Do NOT store:
- one-off trivia
- temporary wording
- raw chat transcript
- sensitive secrets unless the user clearly wants them remembered

Write a concise memory summary in plain text, max 900 characters.
""".strip()

    response = client.responses.create(
        model=MEMORY_MODEL,
        instructions="You maintain compact long-term conversational memory.",
        input=prompt,
        max_output_tokens=300,
    )

    new_memory = (getattr(response, "output_text", "") or "").strip()
    if new_memory:
        state["long_memory"] = new_memory[:900]
        save_db()


def transcribe_voice_sync(file_path: Path) -> str:
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model=TRANSCRIBE_MODEL,
            file=audio_file,
        )

    text = getattr(transcript, "text", "").strip()
    if not text:
        raise ValueError("Не удалось распознать голосовое сообщение")

    return text


def synthesize_voice_sync(text: str, out_path: Path):
    voice_text = text[:4096]

    with client.audio.speech.with_streaming_response.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=voice_text,
        instructions="Speak naturally, clearly, confidently, with expressive human warmth and light playful attitude.",
        response_format="opus",
    ) as response:
        response.stream_to_file(out_path)


async def generate_reply(user_id: int, user_text: str) -> str:
    return await asyncio.to_thread(generate_reply_sync, user_id, user_text)


async def refresh_long_memory(user_id: int):
    await asyncio.to_thread(refresh_long_memory_sync, user_id)


async def transcribe_voice(file_path: Path) -> str:
    return await asyncio.to_thread(transcribe_voice_sync, file_path)


async def synthesize_voice(text: str, out_path: Path):
    await asyncio.to_thread(synthesize_voice_sync, text, out_path)


# =========================
# SENDERS
# =========================
async def send_text_reply(message: Message, text: str):
    for chunk in split_text(text):
        await message.answer(chunk)


async def send_voice_reply(message: Message, text: str):
    out_path = TEMP_DIR / f"reply_{message.chat.id}_{message.message_id}.ogg"

    try:
        await synthesize_voice(text, out_path)
        voice_file = FSInputFile(str(out_path))
        await message.answer_voice(voice=voice_file, caption="AI voice")
    finally:
        if out_path.exists():
            try:
                out_path.unlink()
            except Exception:
                logging.exception("Не удалось удалить временный voice-файл")


# =========================
# CORE CHAT LOGIC
# =========================
async def process_user_message(message: Message, user_text: str, source: str = "text"):
    user_id = message.from_user.id
    state = get_user_state(user_id)

    user_text = normalize_text(user_text)

    if not user_text:
        await message.answer("Слова закончились? Трагично.")
        return

    state["history"].append({"role": "user", "content": user_text})
    state["history"] = trim_history(state["history"], 20)
    state["turns_since_memory_refresh"] = state.get("turns_since_memory_refresh", 0) + 1
    save_db()

    try:
        reply = await generate_reply(user_id, user_text)
    except Exception as e:
        logging.exception("Ошибка генерации ответа")
        await message.answer(f"Я словила ошибку у OpenAI: {e}")
        return

    state["history"].append({"role": "assistant", "content": reply})
    state["history"] = trim_history(state["history"], 20)
    save_db()

    await send_text_reply(message, reply)

    # если пользователь прислал voice — логично ответить voice
    # или можно включить постоянный голос командой /voice_on
    if source == "voice" or state.get("voice_enabled", False):
        try:
            await send_voice_reply(message, reply)
        except Exception:
            logging.exception("Ошибка TTS")
            await message.answer("С голосом сейчас что-то перекосило, но текст я уже отправила.")

    # обновляем долгую память не на каждом сообщении, чтобы не жрать лишние токены
    if state.get("turns_since_memory_refresh", 0) >= 4:
        try:
            await refresh_long_memory(user_id)
            state["turns_since_memory_refresh"] = 0
            save_db()
        except Exception:
            logging.exception("Ошибка обновления долгой памяти")


# =========================
# COMMANDS
# =========================
@dp.message(CommandStart())
async def start(message: Message):
    text = (
        "Ну давай. Посмотрим, есть ли там интеллект.\n\n"
        "Команды:\n"
        "/voice_on — включить голосовые ответы\n"
        "/voice_off — выключить голосовые ответы\n"
        "/clear_memory — очистить память\n\n"
        "И да: голос тут синтетический, AI-generated."
    )
    await message.answer(text)


@dp.message(Command("voice_on"))
async def voice_on(message: Message):
    state = get_user_state(message.from_user.id)
    state["voice_enabled"] = True
    save_db()
    await message.answer("Голос включен. Теперь я буду ещё и звучать. Какая щедрость.")


@dp.message(Command("voice_off"))
async def voice_off(message: Message):
    state = get_user_state(message.from_user.id)
    state["voice_enabled"] = False
    save_db()
    await message.answer("Голос выключен. Придётся читать глазами. Держись.")


@dp.message(Command("clear_memory"))
async def clear_memory(message: Message):
    state = get_user_state(message.from_user.id)
    state["history"] = []
    state["long_memory"] = ""
    state["turns_since_memory_refresh"] = 0
    save_db()
    await message.answer("Память очищена. Начинаем с чистого листа, как после неловкого позора.")


# =========================
# VOICE HANDLER
# =========================
@dp.message(F.voice)
async def handle_voice(message: Message):
    in_path = TEMP_DIR / f"input_{message.chat.id}_{message.message_id}.ogg"

    try:
        file_info = await message.bot.get_file(message.voice.file_id)
        await message.bot.download(file_info, destination=in_path)

        user_text = await transcribe_voice(in_path)

        # по желанию можно показать распознанный текст:
        # await message.answer(f"🎤 {user_text}")

        await process_user_message(message, user_text, source="voice")

    except Exception as e:
        logging.exception("Ошибка обработки голосового сообщения")
        await message.answer(f"Голосовое споткнулось об реальность: {e}")

    finally:
        if in_path.exists():
            try:
                in_path.unlink()
            except Exception:
                logging.exception("Не удалось удалить временный входной аудио-файл")


# =========================
# TEXT HANDLER
# =========================
@dp.message(F.text)
async def handle_text(message: Message):
    await process_user_message(message, message.text, source="text")


# =========================
# MAIN
# =========================
async def main():
    logging.info("Бот запущен...")
    bot = Bot(token=TELEGRAM_TOKEN)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())