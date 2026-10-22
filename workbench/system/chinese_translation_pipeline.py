"""Chinese ASR snapshots -> guarded Chinese refinement -> foreign translation."""
import asyncio
from dataclasses import dataclass
import time

from .adaptive_translation import AdaptiveTranslationSnapshots
from .refinement_guard import join_refined_segments
from .translation_previews import call_with_previews
from .translation_scheduler import LatestSlot, SourceSnapshot, thread_call
from .translation_windows import StableTranslationSnapshots
from .window_refinement import StreamingRefinementDisplay


@dataclass(frozen=True)
class RefinedSnapshot:
    source: SourceSnapshot
    payload: dict
    text: str
    translation_revision: int
    final: bool


def refined_view(result, source, revision, *, final=False):
    """Expose raw pending Chinese, but select only guarded spans for MT."""
    if final:
        clean = str(result.get('clean_text', ''))
        return {**result, 'clean_text': clean, 'display_text': clean,
                'display_refined_text': clean, 'committed_clean_text': clean,
                'active_clean_text': '', 'pending_raw_text': '',
                'has_pending_refinement': False, 'display_revision': revision}
    committed = []
    for span in result.get('refinement_source_spans', ()):
        if span.get('state') != 'committed':
            break
        committed.append(span)
    view = StreamingRefinementDisplay()
    if committed:
        view.accept({'raw_text': ''.join(span['source_text'] for span in committed),
                     'clean_text': join_refined_segments(span['clean_text'] for span in committed),
                     'refinement_source_spans': committed}, revision)
    display = view.compose(source)
    return {**result, **display, 'clean_text': display['display_text'],
            'display_revision': revision}


async def run_chinese_translation_pipeline(stream, session, translate, publish, *,
                                            target_language, stream_translate=None,
                                            strategy='adaptive'):
    """Bound both pending stages and translate only completed Chinese results.

    Group callbacks release guarded prefixes to MT while the Refiner continues
    its later groups. ASR revisions and translation revisions are independent.
    """
    sources, refined = LatestSlot(), LatestSlot()
    stats = {'source_updates': 0, 'refinement_updates': 0, 'translation_updates': 0,
             'discarded_translation_results': 0, 'discarded_refinement_results': 0}
    revision = 0
    translation_revision = 0
    last_queued = None
    translator = (AdaptiveTranslationSnapshots(translate, stream_translate,
                                               preview_confirmed=True)
                  if strategy == 'adaptive' else
                  StableTranslationSnapshots(translate, stream_translate=stream_translate))

    async def produce():
        nonlocal revision
        previous = None
        finished = False
        async for source, final, progress in stream:
            if finished:
                raise RuntimeError('Source emitted text after final')
            if not isinstance(source, str) or not source.strip():
                if final:
                    raise ValueError('没有识别到中文语音，请检查音频文件。')
                await publish({'event': 'progress', **progress})
                continue
            if source == previous and not final:
                await publish({'event': 'progress', **progress})
                continue
            revision += 1
            previous, finished = source, final
            stats['source_updates'] += 1
            item = SourceSnapshot(revision, source, final, dict(progress))
            await publish({'event': 'source', 'revision': revision,
                           'source_text': source, 'source_language': 'zh', **progress})
            await sources.put(item)
        if not finished:
            raise ValueError('输入流结束但未发送final，无法确认最终文本。')

    async def offer(snapshot, payload, *, final=False):
        nonlocal translation_revision, last_queued
        # Never substitute an ASR/pending tail when no guarded prefix exists.
        text = payload.get('clean_text', '') if final else payload.get('committed_clean_text', '')
        if not final and (not text or text == last_queued):
            return
        translation_revision += 1
        last_queued = text
        await refined.put(RefinedSnapshot(snapshot, payload, text,
                                         translation_revision, final))

    async def refinement_worker():
        while True:
            snapshot = await sources.get()
            await publish({'event': 'refining', 'revision': snapshot.revision,
                           'refinement_revision': snapshot.revision})
            started = time.perf_counter()

            def decorate(result, *, final=False):
                view = refined_view(result, snapshot.source, snapshot.revision, final=final)
                return {**view, 'revision': snapshot.revision,
                        'refinement_revision': snapshot.revision,
                        'source_text': snapshot.source, 'source_language': 'zh',
                        'target_language': target_language, 'input_kind': 'chinese_asr_snapshot',
                        'refinement_ms': (time.perf_counter()-started)*1000,
                        **snapshot.progress}

            async def group_done(result):
                payload = decorate(result)
                await publish({**payload, 'event': 'refinement_progress'})
                await offer(snapshot, payload)

            from .offline_text_refinement import OfflineTextRefinementSession
            if isinstance(session, OfflineTextRefinementSession):
                result = await call_with_previews(
                    lambda callback: session.update(snapshot.source, revision=snapshot.revision,
                                                    final=snapshot.final, on_progress=callback),
                    group_done)
            else:
                result = await thread_call(session.update, snapshot.source,
                                           revision=snapshot.revision, final=snapshot.final)
            payload = decorate(result, final=snapshot.final)
            stats['refinement_updates'] += 1
            await publish({**payload, 'event': 'refined'})
            await offer(snapshot, payload, final=snapshot.final)
            if snapshot.final:
                return

    async def translation_worker():
        while True:
            item = await refined.get()
            sequence = 0

            async def preview(text):
                nonlocal sequence
                sequence += 1
                await publish({'event': 'translation_preview',
                    'revision': item.source.revision,
                    'translation_revision': item.translation_revision,
                    'translation_source_text': item.text,
                    'source_language': 'zh', 'target_language': target_language,
                    'translation': text, 'preview_sequence': sequence, 'provisional': True})

            def update(callback=None):
                if isinstance(translator, AdaptiveTranslationSnapshots):
                    return translator.update(item.text, target_language,
                                             on_partial=callback, final=True)
                return translator.update(item.text, target_language, on_partial=callback)

            if item.text.strip():
                if stream_translate:
                    text, elapsed = await call_with_previews(update, preview)
                else:
                    text, elapsed = await thread_call(update)
                translation_stats = {**translator.last_stats,
                                     'strategy': 'refined_context_windows_v1'}
            else:
                text, elapsed = '', 0.0
                translation_stats = dict(translator.last_stats)
            stats['translation_updates'] += 1
            event = {'event': 'final' if item.final else 'translation',
                     'revision': item.source.revision,
                     'translation_revision': item.translation_revision,
                     'source_text': item.source.source, 'source_language': 'zh',
                     'target_language': target_language,
                     'translation_source_text': item.text, 'translation': text,
                     'translation_ms': elapsed, 'translation_stats': translation_stats,
                     **item.source.progress}
            if item.final:
                event = {**item.payload, **event,
                         'pipeline_stats': {**stats,
                             'merged_source_snapshots': sources.replaced,
                             'merged_refinement_snapshots': refined.replaced,
                             'max_pending_per_stage': 1}}
            await publish(event)
            if item.final:
                return

    tasks = [asyncio.create_task(produce()), asyncio.create_task(refinement_worker()),
             asyncio.create_task(translation_worker())]
    joined = asyncio.gather(*tasks)
    try:
        await asyncio.shield(joined)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if joined.done() and not joined.cancelled():
            joined.exception()
