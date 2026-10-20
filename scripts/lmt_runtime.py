"""Pinned LMT inference with the author's translation prompt and chat template."""
import json
from pathlib import Path
from time import perf_counter

LANGUAGES = {'en':'English','fr':'French','de':'German','es':'Spanish',
             'ru':'Russian','ja':'Japanese','ko':'Korean','ar':'Arabic'}


class LmtRuntime:
    def __init__(self, path, gpu, backend='ctranslate2', batch_size=1, memory_fraction=.07):
        import transformers
        self.path, self.gpu, self.backend = Path(path), gpu, backend
        self.batch_size = batch_size
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(path, local_files_only=True, padding_side='left')
        self.special = set(self.tokenizer.all_special_ids)
        self.config = json.loads((self.path/'config.json').read_text())
        generation = json.loads((self.path/'generation_config.json').read_text())
        self.eos_ids = generation['eos_token_id']
        if isinstance(self.eos_ids, int):
            self.eos_ids = [self.eos_ids]
        self.parameter_count = None
        if backend == 'ctranslate2':
            import ctranslate2
            self.model = ctranslate2.Generator(str(self.path/'ct2-int8'),device='cuda',device_index=gpu,
                compute_type='int8_float16',inter_threads=1,intra_threads=2)
        else:
            import torch
            torch.set_num_threads(2)
            torch.cuda.set_device(gpu)
            torch.cuda.set_per_process_memory_fraction(memory_fraction,gpu)
            self.model = transformers.AutoModelForCausalLM.from_pretrained(path,local_files_only=True,
                dtype=torch.float16,attn_implementation='sdpa',low_cpu_mem_usage=True).eval().to(f'cuda:{gpu}')
            self.parameter_count = sum(p.numel() for p in self.model.parameters())

    def prompt_ids(self, source, language):
        name = LANGUAGES[language]
        prompt = f'Translate the following text from {name} into Chinese:\n{name}: {source}\nChinese:'
        return self.tokenizer.apply_chat_template([{'role':'user','content':prompt}],
            tokenize=True,add_generation_prompt=True,return_dict=False)

    def translate(self, texts, language, committed=None, beams=3, max_new_tokens=256, callback=None, streamer=None):
        t0 = perf_counter()
        committed = list(committed or [])
        prompt_ids = [self.prompt_ids(text,language) for text in texts]
        if any(len(ids)+len(committed)+max_new_tokens>32768 for ids in prompt_ids):
            raise ValueError('LMT context exceeded; no silent source truncation')
        starts = [ids+committed for ids in prompt_ids]
        limit = max(1,max_new_tokens-len(committed))
        if self.backend == 'ctranslate2':
            tokens = [self.tokenizer.convert_ids_to_tokens(ids) for ids in starts]
            assert all(self.tokenizer.convert_tokens_to_ids(ts)==ids for ts,ids in zip(tokens,starts))
            options = dict(beam_size=beams,max_length=limit,include_prompt_in_result=False,
                return_end_token=True,end_token=self.tokenizer.convert_ids_to_tokens(self.eos_ids),
                max_batch_size=self.batch_size)
            if callback is not None:
                options['callback'] = callback
            output = self.model.generate_batch(tokens,**options)
            generated = [self.tokenizer.convert_tokens_to_ids(row.sequences[0]) for row in output]
        else:
            import torch
            assert len(starts)==1,'Native control runs one input at a time'
            ids = torch.tensor(starts,dtype=torch.long,device=f'cuda:{self.gpu}')
            torch.cuda.synchronize(self.gpu)
            with torch.inference_mode():
                generated = self.model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),
                    num_beams=beams,do_sample=False,max_new_tokens=limit,use_cache=True,
                    streamer=streamer,
                    eos_token_id=self.eos_ids,pad_token_id=self.tokenizer.pad_token_id)
            torch.cuda.synchronize(self.gpu)
            generated = [row[len(starts[0]):].tolist() for row in generated]
        elapsed = perf_counter()-t0
        results = []
        for prompt, ids in zip(prompt_ids,generated):
            body = committed+[i for i in ids if i not in self.special]
            text = self.tokenizer.decode(body,skip_special_tokens=True)
            results.append({'hypothesis':text,'body_ids':body,'source_tokens':len(prompt),
                'target_tokens':len(body),'unknown_tokens':sum(i==self.tokenizer.unk_token_id for i in ids),
                'eos':bool(ids and ids[-1] in self.eos_ids),
                'replacement_characters':text.count('\ufffd')})
        return results,elapsed


def complete_unicode_prefix(tokenizer, candidate, length, minimum):
    """Qwen byte tokens can end inside a character; never commit a partial one."""
    while length>minimum and '\ufffd' in tokenizer.decode(candidate[:length],skip_special_tokens=True):
        length-=1
    return length
