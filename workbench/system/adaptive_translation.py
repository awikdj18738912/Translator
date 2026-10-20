"""Context-bearing stable groups with revisable previews and literal spellings."""
import re
import time

from .translation_windows import _sentence_data, _split_long
from .refinement_guard import join_refined_segments


# Preserve explicit letter-by-letter spelling, not arbitrary English words.
SPELLING = re.compile(r'(?<![A-Za-z0-9])(?:[A-Z][ \t]+){2,}[A-Z](?![A-Za-z0-9])')


def context_windows(source, *, max_chars=700, group_sentences=3, min_chars=160):
    """Keep neighboring short sentences together; return (source, stable)."""
    sentences, complete = _sentence_data(source)
    result=[];group=[];size=0;completed=0
    def flush(stable):
        nonlocal group,size,completed
        if group:result.append((''.join(group),stable))
        group=[];size=0;completed=0
    for index,sentence in enumerate(sentences):
        pieces=_split_long(sentence,max_chars)
        if len(pieces)>1:
            flush(True)
            for offset,piece in enumerate(pieces):
                result.append((piece,offset<len(pieces)-1 or index<complete))
            continue
        if group and size+len(sentence)>max_chars:flush(True)
        group.append(sentence);size+=len(sentence)
        if index<complete:completed+=1
        if index<complete and (completed>=group_sentences or size>=min_chars):flush(True)
    flush(False)
    return result


class AdaptiveTranslationSnapshots:
    def __init__(self,translate,stream_translate=None):
        self.translate=translate;self.stream_translate=stream_translate
        self.cache={};self.stable_translation='';self.active_translation=''
        self.last_translation=''
        self.last_stats={};self.totals={'model_calls':0,'preview_model_calls':0,'confirmation_model_calls':0,'source_characters':0}

    def update(self,source,language,on_partial=None,*,final=False):
        if not isinstance(source,str) or not source.strip() or len(source)>20000:
            raise ValueError('原文为空或超过20000字符，请分段处理。')
        start=time.perf_counter();output=[];stable_parts=[];active_parts=[];fresh={}
        stats={'model_calls':0,'preview_model_calls':0,'confirmation_model_calls':0,
               'source_characters':0,'cache_hits':0,'stable_windows':0,'protected_spellings':0}
        def emit(value):
            # Scanning cached windows must not erase a previously visible tail.
            if value!=self.last_translation and self.last_translation.startswith(value):return
            if on_partial:on_partial(value)

        def decode(text,preview,publish):
            parts=[];cursor=0
            spans=list(SPELLING.finditer(text))
            for match in [*spans,None]:
                end=match.start() if match else len(text)
                piece=text[cursor:end]
                if piece.strip():
                    if not any(char.isalnum() for char in piece):
                        if parts:parts[-1]+=piece.strip()
                        else:parts.append(piece.strip())
                    else:
                        stats['model_calls']+=1
                        is_stream=preview and self.stream_translate is not None
                        stats['preview_model_calls' if is_stream else 'confirmation_model_calls']+=1
                        stats['source_characters']+=len(piece)
                        if is_stream:
                            def partial(value):publish(join_refined_segments([*parts,value]))
                            value=self.stream_translate(piece.strip(),language,partial)
                        else:value=self.translate(piece.strip(),language)
                        if not isinstance(value,str) or not value.strip():
                            raise RuntimeError('翻译模型未返回完整中文，请缩短片段后重试。')
                        parts.append(value.strip())
                if match:
                    parts.append(match.group());stats['protected_spellings']+=1
                    cursor=match.end()
                    publish(join_refined_segments(parts))
            return join_refined_segments(parts)

        windows=context_windows(source)
        for text,stable in windows:
            confirmed=stable or final
            key=(language,text.strip(),confirmed)
            value=self.cache.get(key)
            def publish(value):
                emit(join_refined_segments([*output,value]))
            if value is None:
                # Only unfinished context uses a provisional decode. Once it
                # closes, quality-decode that same group once and retain it.
                value=decode(text.strip(),not confirmed,publish)
                fresh[key]=value
            else:
                stats['cache_hits']+=1;fresh[key]=value
            output.append(value)
            (stable_parts if confirmed else active_parts).append(value)
            stats['stable_windows']+=int(confirmed)
            emit(join_refined_segments(output))
        self.cache=fresh
        self.stable_translation=join_refined_segments(stable_parts)
        self.active_translation=join_refined_segments(active_parts)
        for name in self.totals:self.totals[name]+=stats[name]
        self.last_stats={**stats,'windows':len(windows),'strategy':'adaptive_context_v1',
                         **{'total_'+name:value for name,value in self.totals.items()}}
        self.last_translation=join_refined_segments(output)
        return self.last_translation,(time.perf_counter()-start)*1000
