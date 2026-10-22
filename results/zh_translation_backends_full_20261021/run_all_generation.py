#!/usr/bin/env python3
"""Sequential, restartable full-corpus backend experiment on physical GPU1."""
import json, os, signal, subprocess, sys, time, urllib.request
from pathlib import Path

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
PY='/home/aim0/data/conda/envs/agentic-asr/bin/python'
CLIENT='/home/aim0/data/venvs/translation-bench/bin/python'
ENV=dict(os.environ,CUDA_VISIBLE_DEVICES='1',HF_HOME='/home/aim0/data/hf-cache',
 HF_HUB_CACHE='/home/aim0/data/hf-cache/hub',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
 OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',VLLM_WORKER_MULTIPROC_METHOD='spawn',
 VLLM_CACHE_ROOT='/home/aim0/data/cache/vllm-translation',
 TORCHINDUCTOR_CACHE_DIR='/home/aim0/data/cache/torchinductor-translation',
 PYTHONUNBUFFERED='1',VLLM_USE_FLASHINFER_SAMPLER='0')

def stop_server(pid):
    try:
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        assert 'vllm.entrypoints.openai.api_server' in cmd and '--port 8090' in cmd,cmd
        group=os.getpgid(pid)
        assert group==pid, (pid,group)
        os.killpg(group,signal.SIGTERM)
        for _ in range(30):
            if not Path(f'/proc/{pid}').exists():break
            time.sleep(1)
    except FileNotFoundError:pass

def complete(key):
    p=BASE/key/'metadata.json'
    return p.exists() and json.loads(p.read_text()).get('status')=='complete' and (BASE/key/'predictions.jsonl').exists()

def run(command,log):
    with log.open('a') as f:
        result=subprocess.run(command,cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'Exit {result.returncode}: {log}')

def main():
    old=BASE/'bf16_vllm/server_config.json'
    if old.exists():stop_server(json.loads(old.read_text())['pid'])
    time.sleep(8)
    phase=BASE/'generation_phase.txt';stop=BASE/'generation_telemetry.STOP'
    stop.unlink(missing_ok=True)
    telemetry=subprocess.Popen([CLIENT,str(ROOT/'results/zh_translation_quantized_full_20261021/collect_gpu1_telemetry.py'),str(BASE/'generation_gpu1.csv'),str(phase),str(stop),'1'],env=ENV)
    try:
        for key,suffix in [('bf16_vllm',''),('fp8_vllm','-FP8'),('nvfp4_vllm','-FP4'),('bf16_transformers',''),('fp8_transformers','-FP8'),('nvfp4_transformers','-FP4')]:
            if complete(key):
                print('SKIP completed',key,flush=True);continue
            folder=BASE/key;folder.mkdir(exist_ok=True)
            phase.write_text(key+':startup')
            print('START',key,flush=True)
            if key.endswith('_vllm'):
                cmd=[PY,'-m','vllm.entrypoints.openai.api_server','--model','/home/aim0/data/models/ASR/Index-Translate-2B'+suffix,
                 '--served-model-name','Index-Translate-2B','--host','127.0.0.1','--port','8090','--dtype','bfloat16',
                 '--max-model-len','1024','--max-num-seqs','128','--max-num-batched-tokens','8192',
                 '--gpu-memory-utilization','0.90','--language-model-only','--no-enable-prefix-caching','--generation-config','vllm']
                if suffix:cmd+=['--quantization','compressed-tensors','--linear-backend','marlin']
                with (folder/'server.log').open('a') as f:
                    server=subprocess.Popen(cmd,env=ENV,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                (folder/'server_config.json').write_text(json.dumps({'command':cmd,'pid':server.pid,'physical_gpu':1,'environment':{k:ENV[k] for k in ('VLLM_USE_FLASHINFER_SAMPLER','CUDA_VISIBLE_DEVICES')}},indent=2))
                try:
                    deadline=time.time()+1200
                    while time.time()<deadline:
                        if server.poll() is not None:raise RuntimeError(f'Server failed: {folder}/server.log')
                        try:
                            if urllib.request.urlopen('http://127.0.0.1:8090/health',timeout=3).status==200:break
                        except Exception:pass
                        time.sleep(3)
                    else:raise TimeoutError(key+' server startup')
                    phase.write_text(key+':translation')
                    run([CLIENT,str(BASE/'run_vllm_translation.py'),'--model-key',key,'--output',str(folder/'predictions.jsonl'),'--concurrency','128','--max-tokens','512','--timeout','600','--retries','4','--resume'],folder/'translation.log')
                finally:
                    stop_server(server.pid)
                    server.wait(timeout=60)
                    time.sleep(8)
            else:
                phase.write_text(key+':translation')
                run([PY,'-u',str(BASE/'run_transformers_translation.py'),'--model-key',key,'--output',str(folder/'predictions.jsonl'),'--batch-size','128','--max-tokens','512','--resume'],folder/'translation.log')
            assert complete(key),key
            print('COMPLETE',key,flush=True)
        (BASE/'generation_complete.json').write_text(json.dumps({'status':'complete','time':time.time()}))
    finally:
        stop.touch();telemetry.wait(timeout=10)

if __name__=='__main__':main()
