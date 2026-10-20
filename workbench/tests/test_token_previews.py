import asyncio
import threading
from lmt_runtime import TokenTextEmitter
from system.translation_previews import call_with_previews


class ByteTokenizer:
    def decode(self,ids,**kwargs):
        return bytes(ids).decode('utf-8',errors='replace')


def test_emitter_waits_for_complete_unicode_and_skips_special_tokens():
    output=[]
    emitter=TokenTextEmitter(ByteTokenizer(),{999},output.append,interval=0)
    for value in '猫🙂'.encode():
        emitter.append(value)
    emitter.append(999)
    assert output==['猫','猫🙂']
    assert all('\ufffd' not in text for text in output)


def test_emitter_coalesces_callbacks_by_time():
    output=[];clock=[0.0]
    emitter=TokenTextEmitter(ByteTokenizer(),set(),output.append,interval=.08,clock=lambda:clock[0])
    emitter.append(ord('a'))
    clock[0]=.01;emitter.append(ord('b'))
    clock[0]=.08;emitter.append(ord('c'))
    assert output==['a','abc']


def test_slow_preview_consumer_does_not_block_decoder_or_build_a_queue():
    async def scenario():
        first_seen=asyncio.Event();release=asyncio.Event();decoder_done=threading.Event()
        received=[]
        def decode(callback):
            for i in range(2000):callback(str(i))
            decoder_done.set()
            return 'complete'
        async def publish(text):
            received.append(text);first_seen.set();await release.wait()
        job=asyncio.create_task(call_with_previews(decode,publish))
        await asyncio.wait_for(first_seen.wait(),2)
        assert await asyncio.wait_for(job,2)=='complete'
        assert decoder_done.is_set()
        assert len(received)==1
    asyncio.run(scenario())
