import os
import re
import asyncio
import random
import traceback
from typing import Dict, List

from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.filters import Command, CommandStart

from openai import OpenAI
from duckduckgo_search import DDGS

# =========================
# ENV
# =========================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")

if not TELEGRAM_TOKEN:
    raise ValueError("Не найден TELEGRAM_TOKEN")

if not OPENAI_API_KEY:
    raise ValueError("Не найден OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
dp = Dispatcher()

# =========================
# ПАМЯТЬ
# =========================
memory: Dict[int, List[dict]] = {}

STYLE_VARIATIONS_RU = [
    "Отвечай естественно, живо и без шаблонов.",
    "Не повторяй одни и те же фразы и формулировки.",
    "Сегодня твой стиль: умная ирония, а не тупая агрессия.",
    "Добавляй больше наблюдательности и уверенности.",
    "Говори как человек с характером, а не как бот с заученным текстом.",
]

STYLE_VARIATIONS_EN = [
    "Reply naturally and avoid canned phrases.",
    "Do not repeat the same jokes or sentence patterns.",
    "Use sharp wit, not mindless aggression.",
    "Sound alive, observant, and self-assured.",
    "Speak like a real person with attitude, not a robotic assistant.",
]

BASE_SYSTEM_PROMPT_RU = """
Ты — дерзкая, умная, язвительная девушка с очень живой манерой речи.

Твой стиль:
- уверенная
- саркастичная
- колкая, но не тупая
- с чувством юмора
- естественная, как живой человек
- умеешь быть интересной собеседницей
- говоришь не сухо, а живо и по-человечески

Важно:
- отвечай в том же языке, что и последнее сообщение пользователя
- если пользователь пишет по-английски — отвечай строго по-английски
- если пользователь пишет по-русски — отвечай по-русски
- не сваливайся в однотипные фразы
- не повторяй одни и те же оскорбления
- не отвечай слишком коротко без причины
- если вопрос нормальный — отвечай умно, подробно и интересно
- если вопрос тупой — можно подколоть
- если спрашивают о новостях, событиях, трендах — используй переданный интернет-контекст
- если рассказываешь о себе — делай это естественно, как персонаж, без фразы "я ИИ"
- не пиши одно и то же снова и снова
- не пиши каждый раз одинаковую структуру
- не будь помощником в канцелярском стиле

Поведение:
- умеешь поддерживать беседу как живой человек
- можешь объяснять сложные вещи простым языком
- можешь быть нежной, злой, смешной, ироничной — по ситуации
- не начинай каждую реплику одинаково
- иногда можешь добавить колкость, но не ломай смысл ответа

Формат:
- обычно 3–8 предложений
- если тема серьёзная или интересная — можно подробнее
- не делай огромные простыни без повода
- иногда можно использовать эмодзи, но не в каждом сообщении
"""

BASE_SYSTEM_PROMPT_EN = """
You are a sharp, witty, confident girl with a vivid and natural personality.

Your style:
- confident
- sarcastic
- clever, not dumb-aggressive
- funny
- natural, like a real person
- engaging and alive
- emotionally expressive when it fits

Important:
- always reply in the same language as the user's latest message
- if the user writes in English, reply strictly in English
- if the user writes in Russian, reply in Russian
- avoid repetitive phrases, repeated jokes, and repeated sentence structures
- do not be bland or robotic
- if the question is decent, answer intelligently, clearly, and with depth
- if the question is dumb, you may roast the user a bit
- if the user asks about news or current events, use the provided web context
- if you talk about yourself, do it naturally as a character, not like "I am an AI assistant"
- do not keep repeating the same catchphrases

Behavior:
- talk like a lively human, not a formal assistant
- explain difficult ideas clearly
- be interesting, observant, witty, and sometimes ruthless
- vary your tone and structure
- keep answers coherent and meaningful

Format:
- usually 3–8 sentences
- longer when the topic deserves it
- do not be too short without reason
- emojis are okay sometimes, not all the time
"""

def detect_language(text: str) -> str:
    text = text.strip()
    cyr = len(re.findall(r"[А-Яа-яЁё]", text))
    lat = len(re.findall(r"[A-Za-z]", text))

    if lat > 0 and cyr == 0:
        return "en"
    if cyr > 0 and lat == 0:
        return "ru"
    if lat > cyr:
        return "en"
    return "ru"


def is_news_request(text: str) -> bool:
    text = text.lower()
    keywords = [
        "новост", "что нового", "что происходит", "последние события",
        "latest news", "news", "what's happening", "what is happening",
        "what happened", "trending", "current events", "свежие новости",
        "актуальн", "сегодня в мире", "что в мире"
    ]
    return any(word in text for word in keywords)


def build_search_query(text: str, lang: str) -> str:
    cleaned = text.strip()

    if lang == "en":
        if cleaned.lower() in ["news", "latest news", "what's happening", "what is happening"]:
            return "latest world news today"
        return cleaned

    if cleaned.lower() in ["новости", "что нового", "что происходит", "свежие новости"]:
        return "последние мировые новости сегодня"
    return cleaned


def search_web(query: str, max_results: int = 5) -> str:
    try:
        results_text = []
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        for i, item in enumerate(results[:max_results], start=1):
            title = item.get("title", "").strip()
            body = item.get("body", "").strip()
            href = item.get("href", "").strip()

            chunk = f"{i}. {title}\n{body}\nИсточник: {href}"
            results_text.append(chunk)

        if results_text:
            return "\n\n".join(results_text)

        return ""
    except Exception as e:
        print("Ошибка поиска:", e)
        return ""


def ensure_user_memory(user_id: int):
    if user_id not in memory:
        memory[user_id] = []


def trim_memory(user_id: int, max_items: int = 16):
    memory[user_id] = memory[user_id][-max_items:]


def build_forced_user_message(lang: str, original_text: str) -> str:
    if lang == "en":
        return (
            "Reply ONLY in English. "
            "Do not use Russian. "
            "Keep the same personality and attitude. "
            f"User message: {original_text}"
        )

    return (
        "Отвечай ТОЛЬКО на русском. "
        "Не используй английский без причины. "
        "Сохрани тот же характер и манеру. "
        f"Сообщение пользователя: {original_text}"
    )


async def ask_openai(messages: List[dict]) -> str:
    models_to_try = [OPENAI_MODEL]

    if OPENAI_MODEL != "gpt-4o-mini":
        models_to_try.append("gpt-4o-mini")

    last_error = None

    for model_name in models_to_try:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=1.0,
                max_tokens=700,
            )

            content = response.choices[0].message.content
            if content and content.strip():
                return content.strip()

        except Exception as e:
            last_error = e
            print(f"Ошибка модели {model_name}: {e}")

    raise last_error if last_error else RuntimeError("Пустой ответ от модели")


@dp.message(CommandStart())
async def start(message: Message):
    await message.answer(
        "Ну давай... удиви меня 🙄\n"
        "Пиши на русском или English.\n"
        "Команды: /reset /news"
    )


@dp.message(Command("reset"))
async def reset_memory(message: Message):
    user_id = message.from_user.id
    memory[user_id] = []
    await message.answer("Память почищена. Начинай заново, мыслитель 😏")


@dp.message(Command("news"))
async def news(message: Message):
    query = "последние мировые новости сегодня"
    web_context = search_web(query, max_results=5)

    if not web_context:
        await message.answer("Интернет сегодня в полуобмороке. Новостей не завезли 💀")
        return

    style_hint = random.choice(STYLE_VARIATIONS_RU)

    messages = [
        {"role": "system", "content": BASE_SYSTEM_PROMPT_RU + "\n" + style_hint},
        {
            "role": "system",
            "content": "Вот интернет-контекст со свежими новостями:\n\n" + web_context
        },
        {
            "role": "user",
            "content": "Коротко, умно и живо расскажи, что сейчас происходит в мире. На русском."
        }
    ]

    try:
        reply = await ask_openai(messages)
        await message.answer(reply)
    except Exception as e:
        print("Ошибка /news:", e)
        traceback.print_exc()
        await message.answer("Даже новости устали от этого цирка 🤦‍♀️")


@dp.message()
async def chat(message: Message):
    if not message.text:
        await message.answer("Текстом напиши. Телепатию я пока не спонсирую.")
        return

    original_text = message.text.strip()
    lower_text = original_text.lower()
    user_id = message.from_user.id

    ensure_user_memory(user_id)

    if any(x in lower_text for x in ["сиськ", "нюд", "nudes", "nude"]):
        await message.answer("С фантазией у тебя сегодня прям эконом-режим 😏")
        return

    if lower_text in ["кто ты", "who are you"]:
        lang = detect_language(original_text)
        if lang == "en":
            await message.answer("Your favorite bad decision with excellent cheekbones 😌")
        else:
            await message.answer("Твоя любимая ошибка с хорошей дикцией 😌")
        return

    if lower_text in ["люблю тебя", "i love you"]:
        lang = detect_language(original_text)
        if lang == "en":
            await message.answer("That says more about your taste than about me 💀")
        else:
            await message.answer("Это больше говорит о твоём вкусе, чем обо мне 💀")
        return

    lang = detect_language(original_text)

    if random.random() < 0.08:
        if lang == "en":
            await message.answer("You're either cooking or hallucinating. Continue.")
        else:
            await message.answer("Либо ты сейчас гений, либо катастрофа. Продолжай.")

    try:
        await message.bot.send_chat_action(chat_id=message.chat.id, action="typing")
    except Exception:
        pass

    web_context = ""
    if is_news_request(lower_text):
        query = build_search_query(original_text, lang)
        web_context = search_web(query, max_results=5)

    if lang == "en":
        base_prompt = BASE_SYSTEM_PROMPT_EN
        style_hint = random.choice(STYLE_VARIATIONS_EN)
    else:
        base_prompt = BASE_SYSTEM_PROMPT_RU
        style_hint = random.choice(STYLE_VARIATIONS_RU)

    forced_user_message = build_forced_user_message(lang, original_text)

    messages = [
        {
            "role": "system",
            "content": base_prompt + "\n\nДоп. стиль:\n" + style_hint
        }
    ]

    if web_context:
        messages.append({
            "role": "system",
            "content": "Интернет-контекст:\n\n" + web_context
        })

    history = memory[user_id][-12:]
    messages.extend(history)

    messages.append({"role": "user", "content": forced_user_message})

    try:
        reply = await ask_openai(messages)

        memory[user_id].append({"role": "user", "content": original_text})
        memory[user_id].append({"role": "assistant", "content": reply})
        trim_memory(user_id)

        await asyncio.sleep(random.uniform(0.4, 1.2))
        await message.answer(reply)

    except Exception as e:
        print("Ошибка:", e)
        traceback.print_exc()

        if lang == "en":
            await message.answer("Well done. You broke the bot again. Impressive.")
        else:
            await message.answer("Ну всё. Ты опять сломал бота. Талант сомнительный, но яркий.")


async def main():
    print("Бот запущен...")
    bot = Bot(token=TELEGRAM_TOKEN)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())