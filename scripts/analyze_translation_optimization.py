"""Replay source text partitions with a fake translator, without model/service startup."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'Codex-web'))
from system.translation_web import source_pieces
from system.translation_windows import StableTranslationSnapshots


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace',type=Path,default=ROOT/'results/codexweb_translation_frontend_20261019/sessions/041624a68768442e8a34f9d1746d77c0.jsonl')
    parser.add_argument('--output',type=Path,default=ROOT/'results/translation_optimization_20261019')
    args=parser.parse_args()
    rows=[json.loads(line) for line in args.trace.read_text().splitlines() if line.strip()]
    updates=[r for r in rows if r.get('event') in ('update','final')]
    assert updates and updates[-1]['event']=='final'
    language=next(r['language'] for r in rows if r.get('event')=='ready')
    old_cache=set();old_calls=old_chars=0
    translator=StableTranslationSnapshots(lambda source,language:'模拟译文')
    for row in updates:
        pieces=source_pieces(row['source_text'])
        for piece in pieces:
            if piece not in old_cache:
                old_calls+=1;old_chars+=len(piece)
        old_cache=set(pieces)
        translator.update(row['source_text'],language)
    actual_asr=sum(r.get('asr_ms',0) for r in rows if r.get('event') in ('source','progress'))
    baseline={'audio_seconds':max(r.get('audio_total_seconds',0) for r in rows),
        'updates':len(updates),'asr_ms':actual_asr,
        'translation_ms':sum(r.get('translation_ms',0) for r in updates),
        'refinement_ms':sum(r.get('refinement_ms',0) for r in updates),
        'refiner_model_calls':sum(r.get('model_calls',0) for r in updates)}
    result={'scope':'CPU source partition replay, deterministic fake translator, every recorded update processed; not an inference speed or translation quality benchmark',
        'services_started':False,'models_loaded':False,'source_trace':str(args.trace),
        'final_source_characters':len(updates[-1]['source_text']),
        'recorded_baseline':baseline,
        'legacy_partition':{'translation_calls':old_calls,'translated_source_characters':old_chars},
        'fixed_three_sentence_groups':{
            'translation_calls':translator.last_stats['total_model_calls'],
            'translated_source_characters':translator.last_stats['total_source_characters']},
        'source_character_reduction_fraction':1-translator.last_stats['total_source_characters']/old_chars,
        'source_trace_sha256':hashlib.sha256(args.trace.read_bytes()).hexdigest(),
        'limitations':['Finer groups increased independent translation calls in this all-updates replay; fixed per-call cost may offset input-length savings.',
            'Latest-slot coalescing reduces processed intermediate revisions under load; actual counts depend on measured ASR/model arrival and compute times.',
            'Changed context segmentation may change translation quality; must revalidate with real models before deployment speed/quality claims.']}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'source_work_comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
