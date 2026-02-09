#!Имопрты
import os
import asyncio
import aiohttp
import time
import logging
import sqlite3
from datetime import datetime, UTC

"""from FaceDetector import FaceDetector"""

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from dotenv import load_dotenv

from maxapi import Bot, Dispatcher
from maxapi.types import Command, BotStarted, MessageCreated, BotRemoved, InputMedia, PhotoAttachmentPayload

#!настройка
logging.basicConfig(level=logging.INFO)

load_dotenv()
TOKEN = os.getenv('TOKEN')
folder = os.getenv('folder')

bot = Bot(TOKEN)
dp = Dispatcher()

SAVE_DIR = "data"
os.makedirs(SAVE_DIR, exist_ok=True)

#!база пользователей
conn = sqlite3.connect("users.db", check_same_thread=False)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    chat_id TEXT PRIMARY KEY,
    user_id TEXT,
    created_at TEXT
)
""")
conn.commit()

def user_exists(chat_id: str) -> bool:
    cur.execute("SELECT 1 FROM users WHERE chat_id=?", (chat_id,))
    return cur.fetchone() is not None


def save_user(chat_id: str, user_id: str):
    cur.execute(
        "INSERT OR IGNORE INTO users VALUES (?, ?, ?)",
        (chat_id, user_id, datetime.now(UTC).isoformat())
    )
    conn.commit()

def delete_user(chat_id: str):
    cur.execute("DELETE FROM users WHERE chat_id=?", (chat_id,))
    conn.commit()


def get_all_chat_ids():
    cur.execute("SELECT chat_id FROM users")
    return [row[0] for row in cur.fetchall()]

#!________________Функции бота
#!Первое сообщение
@dp.bot_started()
async def on_bot_started(event: BotStarted):
    await bot.send_message(
        chat_id=event.chat_id,
        text="Привет! Напиши любое сообщение, если хочешь воспользоваться моими функциями"
    )
#!Для дебага    
"""@dp.message_created()
async def debug(event):
    from pprint import pprint
    pprint(event.model_dump())"""
#!Поведение взависимости от сообщения пользователя
@dp.message_created()
async def on_message(event: MessageCreated):
    text = (event.message.body.text or "").strip()
    print("TEXT:", repr(event.message.body.text))
    if text == ("/stop"):
        chat_id = str(event.message.recipient.chat_id)
        if user_exists(chat_id):
            delete_user(chat_id)
            await bot.send_message(
                chat_id=chat_id,
                text="❌ Ты удалён из базы"
            )
        else:
            await bot.send_message(
                chat_id=chat_id,
                text="ℹ️ Ты и так не был в базе"
            )

        return
    
    if text.startswith("/"):
        return

    chat_id = str(event.message.recipient.chat_id)
    user_id = str(event.message.sender.user_id)

    if not user_exists(chat_id):
        save_user(chat_id, user_id)
        await bot.send_message(
            chat_id=chat_id,
            text="✅ Ты добавлен в базу. Чтобы отключиться, напиши /stop"
        )
    else:
        await bot.send_message(
            chat_id=chat_id,
            text="ℹ️ Ты уже есть в базе"
        )
#!Сообщение с вложением
@dp.message_created()
async def on_message_attachments(event: MessageCreated):
    message = event.message
    chat_id = message.recipient.chat_id
    #проверка на наличие пользователя в базе
    if not user_exists(chat_id):
        await bot.send_message(
            chat_id=chat_id,
            text="⛔ У тебя нет доступа к отправке файлов"
        )
        return
    #проверка на наличие вложений
    if not message.body.attachments:
        return

    for att in message.body.attachments:
        if isinstance(att.payload, PhotoAttachmentPayload):
            url = att.payload.url
            token = att.payload.token

            filename = os.path.join(
                SAVE_DIR,
                f"{chat_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            )

            await download_file(url, token, filename)

            await bot.send_message(
                chat_id=chat_id,
                text="📸 Изображение сохранено"
            )
#!функция уведомления
"""async def send_message(letter: str):
    chat_ids = get_all_chat_ids()
    for chat_id in chat_ids:
        try:
            await bot.send_message(
            chat_id=Chat_id,
            text="Посторонний",
            attachments=[InputMedia(path=letter)]
            )
        except Exception as e:
            logging.warning(f'Не удалось отправить: {e}')
    print("отправлено")"""
#!Вспомогателные функции---------------------------------------
#!Скачивание
async def download_file(url: str, token: str, filename: str):
    headers = {
        "Authorization": f"Bearer {token}"
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Ошибка скачивания: {resp.status}")

            with open(filename, "wb") as f:
                f.write(await resp.read())
#!поиск новых файлов и отправка
class FileHandler1(FileSystemEventHandler):
    def __init__(self, loop):
        self.loop = loop
    def on_created(self, event):
        if not event.is_directory:
            letter = event.src_path
            print("Файл обнаружен", letter)

            for chat_id in get_all_chat_ids():
                asyncio.run_coroutine_threadsafe(
                    bot.send_message(
                        chat_id=chat_id,
                        text="Обнаружен посторонний",
                        attachments=[InputMedia(path=letter)]
                    ),
                    self.loop
                )
            print("Отправлено")
"""#Сохранение доверенных лиц через камеру
@dp.message_created(Command('save'))
async def on_save(event: MessageCreated):
    chat_id = str(event.message.recipient.chat_id)
    #Вызов функции из FaceDetector
    save_face()

    await bot.send_message(
        chat_id=chat_id,
        text="Сохранено"
    )"""
"""#Вызов подсказки
@dp.message_created(Command('help'))
async def on_help(event: MessageCreated):
    chat_id = str(event.message.recipient.chat_id
    await bot.send_message(
        chat_id=chat_id,
        text="Команды:\nsave -> сохранение лица в камере\nstop -> прекращение работы бота"
    )"""
#!Наблюдение за папкой
def watchdog(loop):
    obs = Observer()
    obs.schedule(FileHandler1(loop), folder, recursive = False)
    obs.start()
    print("Отслеживание в", folder)
    return obs
#!main
async def mainbot():
    loop = asyncio.get_running_loop()
    obs = watchdog(loop)
    try:
        await dp.start_polling(
            bot,
            skip_updates=True
            )
    finally:
        obs.stop()
        obs.join()
        conn.close()
