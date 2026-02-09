import asyncio
from threading import Thread
from test3maxbot import mainbot
import FaceDetector as fd


async def main3():
    detector = fd.FaceDetector()
    Thread(target=detector.run(), daemon=True).start()
    await mainbot()

if __name__ == "__main__":
    asyncio.run(main3())
