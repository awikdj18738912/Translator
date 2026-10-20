"""Launch the existing Codex-web UI with optional text/audio translation routes."""
import argparse
from pathlib import Path
import sqlite3
import sys

from lmt_runtime import LmtRuntime


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--codexweb-root',type=Path,default=root.parent/'Codex-web')
    parser.add_argument('--translation-model',type=Path,default=Path('/home/aim0/data/models/ASR/LMT-60-0.6B'))
    parser.add_argument('--refiner-model',type=Path,default=Path('/home/aim0/data/models/ASR/AgenticASR-Refiner'))
    parser.add_argument('--gpu',type=int,default=1)
    parser.add_argument('--port',type=int,default=8081)
    parser.add_argument('--asr-url',default='http://127.0.0.1:8766')
    parser.add_argument('--data-dir',type=Path,default=root/'results/codexweb_translation_frontend_20261019')
    args=parser.parse_args()
    sys.path.insert(0,str(args.codexweb_root.resolve()))
    from system.web_app import create_app
    from system.entity_store import EntityStore
    from system.translation_web import install_translation_routes
    import uvicorn
    import torch
    torch.set_num_threads(4)
    args.data_dir.mkdir(parents=True,exist_ok=True)
    db=args.data_dir/'entities.db'
    original=args.codexweb_root/'data/entities.db'
    if not db.exists() and original.exists():
        with sqlite3.connect(original.resolve().as_uri()+'?mode=ro',uri=True) as src,sqlite3.connect(db) as dst:
            src.backup(dst)
    store=EntityStore(db)
    for canonical,aliases,kind in [('iPhone',('iphone','Iphone'),'PRODUCT'),('SUS304',(),'MODEL'),
                                  ('AcmeFlow',(),'ORG'),('阀门',('阀門',),'TERM'),('不锈钢',('不鏽鋼','不銹鋼'),'TERM')]:
        try:store.get_entity(canonical,'trade')
        except KeyError:
            store.upsert_entity(canonical,aliases=aliases,entity_type=kind,domain='trade',normalization_policy='normalize')
    app=create_app(args.refiner_model,f'cuda:{args.gpu}',args.asr_url,None,256,
        args.data_dir/'original_web.jsonl',entity_db=db,entity_fuzzy_mode='off',refinement_gate_mode='off',
        entity_audio_repair=False,entity_asr_context=False,entity_ctc_mode='off')
    print('Loading LMT translation model...',flush=True)
    lmt=LmtRuntime(args.translation_model,args.gpu)
    def translate(source,language):
        rows,_=lmt.translate([source],language,beams=3,max_new_tokens=512)
        if not rows[0]['eos']:
            raise RuntimeError('翻译超过生成上限，请缩短输入片段。')
        return rows[0]['hypothesis']
    translate('Please confirm the delivery date.','en')
    install_translation_routes(app,translate,asr_url=args.asr_url,output_dir=args.data_dir/'sessions')
    print(f'Translation UI: http://127.0.0.1:{args.port}',flush=True)
    uvicorn.run(app,host='127.0.0.1',port=args.port,log_level='info')


if __name__=='__main__':
    main()
