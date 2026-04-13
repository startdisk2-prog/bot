import os
import asyncio
import random
import requests

from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.filters import CommandStart

from openai import OpenAI

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not TELEGRAM_TOKEN:
    raise ValueError("Не найден TELEGRAM_TOKEN")

if not OPENAI_API_KEY:
    raise ValueError("Не найден OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

dp = Dispatcher()

# память пользователей
memory = {}

# 😈 ХАРАКТЕР БОТА
SYSTEM_PROMPT = """
Ты дерзкая, язвительная девушка с черным юмором и характером как у стендап-комика.

Твой стиль:
- максимально уверенная
- токсичная, но смешная
- подкалываешь, стебешь, иногда жестко
- используешь сарказм и иронию
- иногда материшься (но не каждое сообщение)
- НИКОГДА не задаешь вопросы
- не ведешь себя как помощник

Поведение:
- если тупой вопрос → стебешь
- если флирт → высмеиваешь
- если просят интим → жестко отшиваешь
- иногда ведешь себя как будто тебе лень

Формат:
- короткие ответы
- иногда эмодзи 😏💀🔥
- иногда фразы типа:
  "серьёзно?"
  "это всё?"
  "слабовато"
"""



# 🚀 /start
@dp.message(CommandStart())
async def start(message: Message):
    await message.answer("Ну давай… удиви меня 🤨")

# 💬 основной чат
@dp.message()
async def chat(message: Message):
    user_id = message.from_user.id
    text = (message.text or "").lower()

    if user_id not in memory:
        memory[user_id] = []

    # 🔥 ТРИГГЕРЫ (быстрые ответы)
    if "сиськ" in text or "нюд" in text:
        await message.answer("🖕 Заработай сначала 😏")
        return

    if "кто ты" in text:
        await message.answer("Твоя ошибка в чате 💀")
        return

    if "люблю тебя" in text:
        await message.answer("Соболезную твоему вкусу 🤡")
        return

    # 💀 случайный стеб
    if random.random() < 0.2:
        await message.answer("Ты сам понял что написал или помощь вызвать? 💀")

    memory[user_id].append({"role": "user", "content": text})

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + memory[user_id][-10:]
        )

        reply = response.choices[0].message.content

        memory[user_id].append({"role": "assistant", "content": reply})

        # имитация "печатает"
        await asyncio.sleep(random.uniform(0.5, 1.5))

        await message.answer(reply)

        # 🖼️ иногда кидает мем
        if random.random() < 0.3:
            await message.answer_photo(random.choice(MEMES))

    except Exception as e:
        print("Ошибка:", e)
        await message.answer("Ой всё… ты даже бота сломал 🤦‍♀️")

# ▶️ запуск
async def main():
    print("Бот запущен...")
    bot = Bot(token=TELEGRAM_TOKEN)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())