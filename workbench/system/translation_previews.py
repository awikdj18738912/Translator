"""Bounded bridge between decoder threads and the WebSocket event loop."""
import asyncio
from .translation_scheduler import thread_call


async def call_with_previews(function,publish):
    loop=asyncio.get_running_loop()
    queue=asyncio.Queue(maxsize=1)
    closed=False

    def offer(text):
        if closed:
            return
        if queue.full():
            queue.get_nowait()
        queue.put_nowait(text)

    def callback(text):
        if not closed:
            loop.call_soon_threadsafe(offer,text)

    async def pump():
        while True:
            text=await queue.get()
            await publish(text)

    task=asyncio.create_task(pump())
    try:
        result=await thread_call(function,callback)
        if task.done():
            task.result()
        return result
    finally:
        closed=True
        task.cancel()
        await asyncio.gather(task,return_exceptions=True)
