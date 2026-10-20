"""Fetch official FLORES-101, retain aligned devtest data and a reproducible manifest."""
import hashlib
import json
from pathlib import Path
import random
import tarfile
import urllib.request


def main():
    root=Path("data/eval/flores101")
    root.mkdir(parents=True,exist_ok=True)
    url="https://dl.fbaipublicfiles.com/flores101/dataset/flores101_dataset.tar.gz"
    archive=root/"flores101_dataset.tar.gz"
    if not archive.exists():
        with urllib.request.urlopen(url,timeout=120) as source, archive.with_suffix('.partial').open('wb') as out:
            while chunk:=source.read(1024*1024):
                out.write(chunk)
        archive.with_suffix('.partial').replace(archive)
    languages={"en":"eng","fr":"fra","de":"deu","es":"spa","ru":"rus","ja":"jpn","ko":"kor","ar":"ara"}
    contents={}
    with tarfile.open(archive,'r:gz') as tar:
        for code in [*languages.values(),"zho_simpl"]:
            matches=[m for m in tar.getmembers() if m.name.endswith('/devtest/'+code+'.devtest')]
            assert len(matches)==1,(code,[m.name for m in matches])
            member=matches[0]
            contents[code]=tar.extractfile(member).read().decode('utf-8').splitlines()
            (root/(code+'.devtest')).write_text('\n'.join(contents[code])+'\n')
        metadata_members=[m for m in tar.getmembers() if 'devtest' in m.name and m.name.endswith('.tsv')]
        metadata=[{'name':m.name,'bytes':m.size} for m in metadata_members]
    assert {len(lines) for lines in contents.values()}=={1012}
    indices=sorted(random.Random(20261017).sample(range(1012),100))
    rows=[]
    for language, code in languages.items():
        for index,(source,target) in enumerate(zip(contents[code],contents['zho_simpl'])):
            rows.append({'id':index+1,'language':language,'flores_language':code,'source':source,'reference':target,
                         'streaming_subset':index in indices})
    output=root/'aligned_devtest.jsonl'
    output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    manifest={'dataset':'FLORES-101','split':'devtest','official_url':url,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
              'aligned_jsonl_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'languages':languages,
              'target':'zho_simpl','sentences_per_language':1012,'total_pairs':len(rows),
              'streaming_subset_per_language':100,'selection':'random.Random(20261017).sample(range(1012),100), sorted; same aligned ids for every language',
              'streaming_ids':[x+1 for x in indices], 'metadata_files_in_archive':metadata,
              'limitations':'Public text benchmark, not business speech; known public benchmark may have been used in model selection; no held-out confidentiality claim'}
    (root/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
