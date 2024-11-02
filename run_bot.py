from asyncio import run
import logging

from aiogram import Bot, Dispatcher, types
from aiogram.filters.command import Command

from env import API_TOKEN, OPENAI_API, CSV_FILENAME, EMBEDDING_MODEL, GPT_MODEL
from gpt_service import GptService

logging.basicConfig(level=logging.DEBUG)

bot = Bot(token=API_TOKEN)
dp = Dispatcher()


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    """Хэндлер на команду /start"""
    await message.answer("Hi there! Ask me question about Linux distros")


@dp.message()
async def handle_message(msg: types.Message):
    answer = gpt.ask_gpt(msg.text)
    await bot.send_message(msg.from_user.id, answer)


async def main():
    # Запуск процесса поллинга новых апдейтов
    await dp.start_polling(bot)
    # await bot.delete_webhook()

if __name__ == "__main__":
    gpt = GptService(OPENAI_API, CSV_FILENAME, EMBEDDING_MODEL, GPT_MODEL)
    logging.info('GPT Started')
    run(main())