#!Импорты
import os
import asyncio
import aiohttp
import time
import logging
import sqlite3
from datetime import datetime, UTC
from PIL import Image
from io import BytesIO

from camera_notifier import set_notifier

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from dotenv import load_dotenv

from maxapi import Bot, Dispatcher, F
from maxapi.types import Command, BotStarted, MessageCreated, BotRemoved, InputMedia, PhotoAttachmentPayload

#!настройка
logging.basicConfig(level=logging.INFO)

load_dotenv()
TOKEN = os.getenv('TOKEN')
folder = os.getenv('folder')
last_intruder_time = 0
last_alert_time = 0

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
        text="Привет! Нажми или напиши /start, чтобы я добавил тебя в базу и ты мог пользоваться моими функциями. Напиши /help, чтобы узнать больше"
    )
#!Для дебага    
"""@dp.message_created()
async def debug(event):
    from pprint import pprint
    pprint(event.model_dump())"""
#!Поведение взависимости от сообщения пользователя
@dp.message_created()
async def on_message(event: MessageCreated):
    message = event.message
    chat_id = str(message.recipient.chat_id)
    user_id = str(message.sender.user_id)

    #вложения
    if message.body.attachments:
        if not user_exists(chat_id):
            await bot.send_message(
                chat_id=chat_id,
                text="⛔ У тебя нет доступа к отправке файлов"
            )
            return

        for att in message.body.attachments:
            if isinstance(att.payload, PhotoAttachmentPayload):
                await save_photo(att, chat_id)
        return  # ⛔ важно

    #текст
    if not message.body.text:
        return

    text = message.body.text.strip()

        #команда help
    if text == "/help":
        await bot.send_message(
            chat_id=chat_id,
            text=(
                "📌 *Справка*\n\n"
                "❓ /help - показать это сообщение\n"
                "💨 /start - добавление в базу\n"
                "⛔ /stop - удалиться из базы, чтобы прекратить пользоваться ботом\n"
                "📷 Отправь фото - я сохраню его, как проверенное лицо\n"
            )
        )
        return

    #команда stop
    if text == "/stop":
        if user_exists(chat_id):
            delete_user(chat_id)
            await bot.send_message(chat_id=chat_id, text="❌ Ты удалён из базы")
        else:
            await bot.send_message(chat_id=chat_id, text="ℹ️ Ты и так не был в базе")
        return

    #команда start
    if text == "/start":
        if not user_exists(chat_id):
            save_user(chat_id, user_id)
            await bot.send_message(
                chat_id=chat_id,
                text="✅ Ты добавлен в базу. Чтобы отключиться, напиши или нажми /stop"
            )
        else:
            await bot.send_message(
                chat_id=chat_id,
                text="ℹ️ Ты уже есть в базе"
            )
    #другие команды
    if text.startswith("/"):
        return
#Вложения
'''@dp.message_created(F.message.attachments)
async def on_message_attachments(event: MessageCreated):
    message = event.message
    chat_id = message.recipient.chat_id

    if not message.body.attachments:
        return

    if not user_exists(str(chat_id)):
        await bot.send_message(
            chat_id=chat_id,
            text="⛔ У тебя нет доступа к отправке файлов"
        )
        return

    for att in message.body.attachments:
        if isinstance(att.payload, PhotoAttachmentPayload):
            await save_photo(att, chat_id)'''
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
async def send_camera_alert():
    global last_alert_time
    now = time.time()
    if now - last_alert_time < 10:
        return
    last_alert_time = now
    for chat_id in get_all_chat_ids():
        try:
            await bot.send_message(
                chat_id=chat_id,
                text="🚨 Камера заблокирована!"
            )
        except Exception as e:
            logging.warning(f"Не отправлено {chat_id}: {e}")
#!Вспомогателные функции---------------------------------------
#!Скачивание
async def download_file(url: str, token: str) -> tuple[bytes, str]:
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "Mozilla/5.0"
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Ошибка скачивания: {resp.status}")

            content_type = resp.headers.get("Content-Type", "")
            data = await resp.read()

    return data, content_type
#!Сохранение фото               
async def save_photo(att, chat_id):
    url = att.payload.url
    token = att.payload.token

    data, content_type = await download_file(url, token)

    # Загружаем изображение из памяти
    image = Image.open(BytesIO(data)).convert("RGB")

    filename = os.path.join(
        SAVE_DIR,
        f"{chat_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    )

    # Сохраняем уже как JPG
    image.save(filename, format="JPEG", quality=95, subsampling=0)

    await bot.send_message(
                chat_id=chat_id,
                text="✅ Успешно сохранено"
            )

    print("Saved as JPG:", filename)
    
#!поиск новых файлов и отправка
class FileHandler1(FileSystemEventHandler):
    def __init__(self, loop):
        self.loop = loop
    def on_created(self, event):
        global last_intruder_time
        now = time.time()
        if now - last_intruder_time < 5:
            return
        if not event.is_directory:
            letter = event.src_path
            print("Файл обнаружен", letter)

            for chat_id in get_all_chat_ids():
                asyncio.run_coroutine_threadsafe(
                    bot.send_message(
                        chat_id=chat_id,
                        text="💢 Обнаружен посторонний",
                        attachments=[InputMedia(path=letter)]
                    ),
                    self.loop
                )
            last_intruder_time = now
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
#обработчик
def camera_blocked_handler():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = camera_blocked_handler._main_loop
    asyncio.run_coroutine_threadsafe(send_camera_alert(), loop)
#!main
async def mainbot():
    loop = asyncio.get_running_loop()

    camera_blocked_handler._main_loop = loop
    
    obs = watchdog(loop)
    set_notifier(camera_blocked_handler)
    try:
        await dp.start_polling(
            bot,
            skip_updates=True
            )
    finally:
        obs.stop()
        obs.join()
        conn.close()

