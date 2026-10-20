"""Paired public-corpus evaluation: full input vs incremental retranslation.

All latency numbers are MT text-call measurements, not speech latency. Streaming
prefixes are artificial; no ASR or speech timing is supplied by FLORES-101.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from time import perf_counter

from benchmark_streaming_mt import common_prefix, percentiles, gpu_snapshot, process_gpu_memory
from mt_tokenizers import load_translation_tokenizer, set_source_language, chinese_decoder_prefix, chinese_language_id


def source_prefixes(text):
    # Identical source-character boundaries for both models. Prefer word boundaries
    # where available; Japanese uses complete Unicode characters, never partial bytes.
    words=list(re.finditer(r"\S+\s*",text))
    boundaries=[]
    for part in (1,2,3):
        if len(words)>=4:
            end=words[max(0, math.ceil(len(words)*part/4)-1)].end()
        else:
            end=math.ceil(len(text)*part/4)
        if 0<end<len(text) and end not in boundaries:
            boundaries.append(end)
    boundaries.append(len(text))
    return [text[:end] for end in boundaries]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=['m2m100_418M','small100','nllb200_600M','LMT-60-0.6B'],required=True)
    parser.add_argument('--gpu',type=int,default=1)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--data',type=Path,default=Path('data/eval/flores101/aligned_devtest.jsonl'))
    parser.add_argument('--output',type=Path,default=Path('results/multilingual_mt_accuracy_20261017'))
    parser.add_argument('--languages',default='en,fr,de,es,ru,ja,ko,ar')
    parser.add_argument('--offline-limit',type=int,default=0)
    parser.add_argument('--offline-subset-only',action='store_true',
                        help='Full-input batch-control run on the fixed streaming subset only')
    parser.add_argument('--stream-limit',type=int,default=100)
    parser.add_argument('--max-decoding-length',type=int,default=256)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    import ctranslate2
    import transformers
    import sacrebleu
    from sacrebleu.metrics import BLEU, CHRF
    from opencc import OpenCC
    from transformers import M2M100Tokenizer
    if ctranslate2.get_cuda_device_count()<=args.gpu:
        raise RuntimeError('CUDA unavailable; no CPU fallback')
    model_path=Path('/home/aim0/data/models/ASR')/args.model
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((model_path/'DOWNLOAD_MANIFEST.json').read_text())
    is_lmt=args.model=='LMT-60-0.6B'
    if is_lmt:
        from lmt_runtime import LmtRuntime,complete_unicode_prefix
        lmt=LmtRuntime(model_path,args.gpu,batch_size=args.batch_size)
        tokenizer,model=lmt.tokenizer,lmt.model
    else:
        tokenizer=load_translation_tokenizer(model_path,args.model)
        model=ctranslate2.Translator(str(model_path/'ct2-int8'),device='cuda',device_index=args.gpu,
                                   compute_type='int8_float16',inter_threads=1,intra_threads=2)
    special=set(tokenizer.all_special_ids)
    lang_prefix=[] if is_lmt else chinese_decoder_prefix(tokenizer,args.model)
    eos=tokenizer.eos_token_id
    normalizer=OpenCC('t2s')
    bleu=BLEU(tokenize='zh')
    chrf=CHRF(word_order=2)
    start_time=perf_counter()
    source_rows=[json.loads(line) for line in args.data.read_text().splitlines()]
    languages=args.languages.split(',')
    data_hash=hashlib.sha256(args.data.read_bytes()).hexdigest()
    metadata={'model':args.model,'repo_id':manifest['repo_id'],'revision':manifest['revision'],
        'created_at':datetime.now(timezone.utc).isoformat(),'dataset':'FLORES-101 devtest',
        'data_sha256':data_hash,'gpu_index':args.gpu,'gpu_snapshot_start':gpu_snapshot(),
        'runtime':'CTranslate2','ctranslate2':ctranslate2.__version__,'transformers':transformers.__version__,
        'sacrebleu':sacrebleu.__version__,'compute_type':'int8_float16','beam_size':3,
        'offline_batch_size':args.batch_size,'stream_batch_size':1,
        'offline_subset_only':args.offline_subset_only,
        'max_decoding_length':args.max_decoding_length,
        'streaming_description':'4 artificial source prefixes; unrestricted retranslation and 2-hypothesis agreement + one-token holdback with forced committed prefix; full source on final flush',
        'scope':'Text MT only. Public written corpus, not business speech; no ASR/audio/refiner/real audio timestamps. Scores are not percent correct.',
        'prefix_policy_sha256':hashlib.sha256(source_prefixes.__code__.co_code).hexdigest()}
    if is_lmt:
        metadata.update({'architecture':'Qwen3ForCausalLM','ct2_class':'Generator',
            'prompt_policy':'Author English translation instruction + checkpoint chat template; default thinking flag unchanged; no examples or auxiliary translation',
            'committed_boundary_policy':'One-token holdback, additionally retreat to a complete Unicode character boundary',
            'max_new_tokens':args.max_decoding_length})
    meta_path=args.output/f'{args.model}.metadata.json'
    if args.resume and meta_path.exists():
        old=json.loads(meta_path.read_text())
        for key in ['data_sha256','compute_type','beam_size','max_decoding_length','revision']:
            assert old[key]==metadata[key],f'Resume configuration mismatch: {key}'
    meta_path.write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')

    def encode(text,language):
        if is_lmt:
            return lmt.prompt_ids(text,language)
        set_source_language(tokenizer,args.model,language)
        ids=tokenizer.encode(text)
        if len(ids)>1000:
            raise RuntimeError(f'Source exceeds model context, id count={len(ids)}')
        return ids

    def translate(texts,language,committed=None):
        if is_lmt:
            return lmt.translate(texts,language,committed,beams=3,max_new_tokens=args.max_decoding_length)
        t0=perf_counter()
        source_ids=[encode(t,language) for t in texts]
        source_tokens=[tokenizer.convert_ids_to_tokens(ids) for ids in source_ids]
        for ids,tokens in zip(source_ids,source_tokens):
            if tokenizer.convert_tokens_to_ids(tokens)!=ids:
                raise RuntimeError('Source token/id roundtrip mismatch')
        prefix=lang_prefix+tokenizer.convert_ids_to_tokens(committed or [])
        outputs=model.translate_batch(source_tokens,target_prefix=[prefix for _ in texts] if prefix else None,
            beam_size=3,max_decoding_length=args.max_decoding_length,max_input_length=0,
            max_batch_size=args.batch_size,return_end_token=True)
        elapsed=perf_counter()-t0
        results=[]
        for output,source in zip(outputs,source_ids):
            ids=tokenizer.convert_tokens_to_ids(output.hypotheses[0])
            body=[x for x in ids if x not in special]
            if committed and body[:len(committed)]!=committed:
                raise RuntimeError('Committed prefix violated')
            results.append({'hypothesis':tokenizer.decode(body,skip_special_tokens=True),'body_ids':body,
                            'source_tokens':len(source),'target_tokens':len(body),
                            'unknown_tokens':ids.count(tokenizer.unk_token_id),
                            'eos':bool(ids and ids[-1]==eos)})
        return results,elapsed

    def corpus_metrics(hypotheses,references):
        raw_bleu=bleu.corpus_score(hypotheses,[references])
        raw_chrf=chrf.corpus_score(hypotheses,[references])
        normalized=[normalizer.convert(t) for t in hypotheses]
        normal_refs=[normalizer.convert(t) for t in references]
        norm_bleu=bleu.corpus_score(normalized,[normal_refs])
        norm_chrf=chrf.corpus_score(normalized,[normal_refs])
        return {'sentences':len(hypotheses),'raw':{'BLEU_zh':raw_bleu.score,'chrFpp':raw_chrf.score},
                't2s':{'BLEU_zh':norm_bleu.score,'chrFpp':norm_chrf.score},
                'bleu_signature':str(bleu.get_signature()),'chrf_signature':str(chrf.get_signature()),
                'bleu_details_t2s':str(norm_bleu)}

    audit=[]
    for lang in languages:
        row=next(x for x in source_rows if x['language']==lang)
        ids=encode(row['source'],lang)
        audit.append({'language':lang,'source':row['source'],'ids':ids,
                      'tokens':tokenizer.convert_ids_to_tokens(ids),
                      'source_ids_roundtrip':tokenizer.convert_tokens_to_ids(tokenizer.convert_ids_to_tokens(ids))==ids,
                      'decoder_prefix':lang_prefix,'target_language_id':None if is_lmt else chinese_language_id(tokenizer,args.model)})
    (args.output/f'{args.model}.protocol_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    translate(['Please confirm the delivery date.'],'en')
    metadata['process_gpu_used_mib_after_warmup']=process_gpu_memory()
    off_path=args.output/f'{args.model}.offline.jsonl'
    stream_path=args.output/f'{args.model}.streaming.jsonl'
    offline=[] if not args.resume or not off_path.exists() else [json.loads(s) for s in off_path.read_text().splitlines()]
    streaming=[] if not args.resume or not stream_path.exists() else [json.loads(s) for s in stream_path.read_text().splitlines()]
    done_off={(x['language'],x['id']) for x in offline}
    done_stream={(x['language'],x['id'],x['mode']) for x in streaming}
    with off_path.open('a' if args.resume else 'w') as off_file,stream_path.open('a' if args.resume else 'w') as stream_file:
        for language in languages:
            rows=[r for r in source_rows if r['language']==language]
            if args.offline_subset_only:
                rows=[r for r in rows if r['streaming_subset']]
            if args.offline_limit:
                rows=rows[:args.offline_limit]
            remaining=[r for r in rows if (language,r['id']) not in done_off]
            for start in range(0,len(remaining),args.batch_size):
                batch=remaining[start:start+args.batch_size]
                outputs,elapsed=translate([r['source'] for r in batch],language)
                for row,result in zip(batch,outputs):
                    result.pop('body_ids')
                    item={**row,**result,'batch_seconds':elapsed,'batch_size':len(batch)}
                    offline.append(item)
                    off_file.write(json.dumps(item,ensure_ascii=False)+'\n')
                off_file.flush()
                if start==0 or (start//args.batch_size+1)%20==0:
                    print(json.dumps({'event':'offline_progress','model':args.model,'language':language,
                                      'completed':min(start+len(batch),len(remaining)),'total':len(remaining)},ensure_ascii=False),flush=True)
            print(json.dumps({'event':'offline_language_complete','model':args.model,'language':language,
                              **corpus_metrics([r['hypothesis'] for r in offline if r['language']==language],
                                               [r['reference'] for r in offline if r['language']==language])},ensure_ascii=False),flush=True)
            selected=[r for r in rows if r['streaming_subset']][:args.stream_limit] if args.stream_limit else []
            for item_index,row in enumerate(selected):
                prefixes=source_prefixes(row['source'])
                assert prefixes[-1]==row['source'] and all(b.startswith(a) for a,b in zip(prefixes,prefixes[1:]))
                for mode in ('retranslate','agreement_forced_prefix'):
                    if (language,row['id'],mode) in done_stream:
                        continue
                    previous_ids,committed,previous_text=[],[],''
                    deleted_chars,steps,all_eos=0,[],True
                    for step_index,prefix_source in enumerate(prefixes):
                        final=step_index==len(prefixes)-1
                        predictions,elapsed=translate([prefix_source],language,committed if mode=='agreement_forced_prefix' else None)
                        prediction=predictions[0]
                        candidate=prediction.pop('body_ids')
                        if mode=='agreement_forced_prefix':
                            agreed=common_prefix(previous_ids,candidate)
                            new_length=len(candidate) if final else max(len(committed),len(agreed)-1)
                            if is_lmt and not final:
                                new_length=complete_unicode_prefix(tokenizer,candidate,new_length,len(committed))
                            committed=candidate[:new_length]
                        shared_chars=len(common_prefix(previous_text,prediction['hypothesis']))
                        erased=len(previous_text)-shared_chars
                        deleted_chars+=erased
                        committed_text=tokenizer.decode(committed,skip_special_tokens=True)
                        steps.append({'step':step_index,'source':prefix_source,**prediction,'seconds':elapsed,
                                      'final':final,'committed_text':committed_text,
                                      'deleted_characters':erased})
                        if len(steps)>1 and not committed_text.startswith(steps[-2]['committed_text']):
                            raise RuntimeError('Rendered committed prefix was revised')
                        previous_ids,previous_text=candidate,prediction['hypothesis']
                        all_eos=all_eos and prediction['eos']
                    item={**row,'mode':mode,'hypothesis':previous_text,'steps':steps,'all_eos':all_eos,
                          'normalized_erasure_characters':deleted_chars/max(1,len(previous_text)),
                          'stable_first_step':next((s['step'] for s in steps if s['committed_text']),None)}
                    streaming.append(item)
                    stream_file.write(json.dumps(item,ensure_ascii=False)+'\n')
                    stream_file.flush()
                if item_index==0 or (item_index+1)%10==0:
                    print(json.dumps({'event':'stream_progress','model':args.model,'language':language,
                                      'completed':item_index+1,'total':len(selected)},ensure_ascii=False),flush=True)
            print(json.dumps({'event':'language_complete','model':args.model,'language':language,
                              'elapsed_seconds':perf_counter()-start_time}),flush=True)
            # Save partial summaries after each language, permitting safe resume.
            summary={**metadata,'status':'running','completed_languages':languages[:languages.index(language)+1],
                     'offline_pairs':len(offline),'streaming_trajectories':len(streaming)}
            (args.output/f'{args.model}.progress.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    scores={}
    for language in languages:
        off=[r for r in offline if r['language']==language]
        selected=[r for r in off if r['streaming_subset']][:args.stream_limit]
        ids={r['id'] for r in selected}
        scores[language]={'offline_full':{**corpus_metrics([r['hypothesis'] for r in off],[r['reference'] for r in off]),
                                         'non_eos_outputs':sum(not r['eos'] for r in off),
                                         'empty_outputs':sum(not r['hypothesis'].strip() for r in off),
                                         'unknown_token_outputs':sum(r.get('unknown_tokens',0)>0 or '<unk>' in r['hypothesis'] for r in off)},
                          'offline_stream_subset':corpus_metrics([r['hypothesis'] for r in selected],[r['reference'] for r in selected]) if selected else None}
        for mode in ('retranslate','agreement_forced_prefix'):
            values=[r for r in streaming if r['language']==language and r['mode']==mode and r['id'] in ids]
            if not values:
                continue
            steps=[s for r in values for s in r['steps']]
            baseline={r['id']:r for r in off}
            metric=corpus_metrics([r['hypothesis'] for r in values],[r['reference'] for r in values])
            metric.update({'step_latency_seconds':percentiles([s['seconds'] for s in steps]),
                           'normalized_erasure_characters':percentiles([r['normalized_erasure_characters'] for r in values]),
                           'cases_different_from_full_input':sum(r['hypothesis']!=baseline[r['id']]['hypothesis'] for r in values),
                           'non_eos_updates':sum(not s['eos'] for s in steps),
                           'empty_updates':sum(not s['hypothesis'].strip() for s in steps),
                           'stable_first_step':percentiles([r['stable_first_step'] for r in values if r['stable_first_step'] is not None])})
            scores[language][mode]=metric
    summary={**metadata,'status':'completed','offline_pairs':len(offline),'streaming_trajectories':len(streaming),
             'total_seconds':perf_counter()-start_time,'gpu_snapshot_end':gpu_snapshot(),
             'process_gpu_used_mib_end':process_gpu_memory(),'scores':scores}
    (args.output/f'{args.model}.summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'event':'evaluation_complete','model':args.model,'total_seconds':summary['total_seconds'],
                      'offline_pairs':len(offline),'streaming_trajectories':len(streaming)}),flush=True)


if __name__=='__main__':
    main()
