#!/usr/bin/env python3
"""Full FLORES throughput comparison at the measured Q8_0 VRAM budget."""
from __future__ import annotations
import csv, hashlib, json, os, signal, statistics, subprocess, sys, time, urllib.request
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(os.environ.get('EQUAL_VRAM_OUTPUT_ROOT',str(Path(__file__).resolve().parent)))
Q8_REFERENCE=Path(os.environ.get('EQUAL_VRAM_Q8_REFERENCE',str(OUT)))
RUNS=ROOT/'results/zh_translation_backends_full_20261021'
QUANT=ROOT/'results/zh_translation_quantized_full_20261021'
DATA=ROOT/'data/eval/zh_translation/flores101_devtest.jsonl'
BENCH='/home/aim0/data/venvs/translation-bench/bin/python'
VLLM='/home/aim0/data/conda/envs/agentic-asr/bin/python'
LLAMA='/home/aim0/data/tools/llama.cpp/build-cuda/bin/llama-server'
MODELROOT=Path('/home/aim0/data/models/ASR')
VLLM_KEYS=('bf16_vllm','fp8_vllm','nvfp4_vllm')
Q8_MODEL=Path('/home/aim0/data/models/ASR/Index-Translate-2B-GGUF/Index-Translate-2B.Q8_0.gguf')
GPU_TOTAL_MIB=24576
ENV=dict(os.environ,CUDA_VISIBLE_DEVICES='1',HF_HOME='/home/aim0/data/hf-cache',
 HF_HUB_CACHE='/home/aim0/data/hf-cache/hub',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
 OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',VLLM_WORKER_MULTIPROC_METHOD='spawn',
 VLLM_CACHE_ROOT='/home/aim0/data/cache/vllm-translation',
 TORCHINDUCTOR_CACHE_DIR='/home/aim0/data/cache/torchinductor-translation',
 VLLM_USE_FLASHINFER_SAMPLER='0',PYTHONUNBUFFERED='1')

def save(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def ensure_gpu_idle():
 result=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()
 used=int(result)
 if used>512: raise RuntimeError(f'GPU1 has {used} MiB in use before starting; refusing a mixed-memory run')

def start_collector():
 phase=OUT/'phase.txt';stop=OUT/'telemetry.STOP';stop.unlink(missing_ok=True)
 proc=subprocess.Popen([BENCH,str(QUANT/'collect_gpu1_telemetry.py'),str(OUT/'gpu1_telemetry.csv'),str(phase),str(stop),'1'],env=ENV)
 return phase,stop,proc

def stop_process_group(proc):
 if proc.poll() is not None:return
 cmd=Path(f'/proc/{proc.pid}/cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
 assert ('llama-server' in cmd or 'vllm.entrypoints.openai.api_server' in cmd),cmd
 assert os.getpgid(proc.pid)==proc.pid
 os.killpg(proc.pid,signal.SIGTERM)
 try:proc.wait(timeout=45)
 except subprocess.TimeoutExpired:
  os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)

def launch(cmd,log,env):
 f=log.open('a');p=subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 return p,f

def wait_health(proc,url,log,timeout=1200):
 deadline=time.time()+timeout
 while time.time()<deadline:
  if proc.poll() is not None:raise RuntimeError(f'server exited; see {log}')
  try:
   if urllib.request.urlopen(url+'/health',timeout=3).status==200:return
  except Exception:pass
  time.sleep(2)
 raise TimeoutError(f'server startup timed out; see {log}')

def run_client(cmd,log):
 with log.open('a') as f:
  r=subprocess.run(cmd,cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError(f'client exit={r.returncode}; see {log}')

def load_rows(path):
 with path.open(encoding='utf-8') as f:return [json.loads(line) for line in f if line.strip()]

def memory_during_requests(key,rows):
 import datetime
 starts=[float(r['request_started_at']) for r in rows if r.get('request_started_at') is not None]
 ends=[float(r['request_ended_at']) for r in rows if r.get('request_ended_at') is not None]
 if not starts or not ends:raise RuntimeError(f'{key}: request timestamps absent')
 values=[];start=min(starts);end=max(ends)
 with (OUT/'gpu1_telemetry.csv').open(newline='') as f:
  for sample in csv.DictReader(f):
   if sample['phase']!=key+':translation':continue
   stamp=datetime.datetime.fromisoformat(sample['timestamp_utc']).timestamp()
   if start<=stamp<=end:values.append(int(sample['memory_used_mib']))
 if len(values)<20:raise RuntimeError(f'{key}: only {len(values)} request-window GPU samples')
 values.sort()
 return {'peak_gpu_memory_mib':max(values),'peak_gpu_memory_gib':max(values)/1024,
  'p95_gpu_memory_mib':values[min(len(values)-1,int(.95*(len(values)-1)))],
  'median_gpu_memory_mib':statistics.median(values),'telemetry_samples':len(values)}

def run_q8(phase):
 ensure_gpu_idle();folder=OUT/'q8_0';folder.mkdir(exist_ok=True);phase.write_text('q8_0:startup')
 cmd=[LLAMA,'-m',str(Q8_MODEL),'--host','127.0.0.1','--port','8088','--n-gpu-layers','all',
  '--ctx-size','131072','--parallel','128','--batch-size','8192','--ubatch-size','2048','--flash-attn','on',
  '--reasoning','off','--threads','12','--threads-batch','16','--cache-ram','0','--no-cache-prompt',
  '--no-cache-idle-slots','--ctx-checkpoints','0','--fit','off','--verbosity','1','--no-webui']
 env=dict(ENV,LD_LIBRARY_PATH='/home/aim0/data/venvs/llama-runtime/lib/python3.12/site-packages/nvidia/cu13/lib')
 proc,logf=launch(cmd,folder/'server.log',env)
 save(folder/'server_config.json',{'command':cmd,'pid':proc.pid,'CUDA_VISIBLE_DEVICES':'1'})
 try:
  wait_health(proc,'http://127.0.0.1:8088',folder/'server.log')
  phase.write_text('q8_0:translation')
  run_client([BENCH,str(QUANT/'run_gguf_translation.py'),'--model-key','q8_0','--output',str(folder/'predictions.jsonl'),
   '--base-url','http://127.0.0.1:8088','--concurrency','128','--max-tokens','512','--timeout','600','--retries','4'],folder/'translation.log')
 finally:
  stop_process_group(proc);logf.close();time.sleep(4)
 meta=json.loads((folder/'metadata.json').read_text())
 if meta.get('status')!='complete' or meta.get('rows_written')!=8096:raise RuntimeError('Q8_0 full reference run did not complete')

def q8_memory_budget():
 meta=json.loads((OUT/'q8_0/metadata.json').read_text());rows=load_rows(OUT/'q8_0/predictions.jsonl')
 start=min(float(r['request_started_at']) for r in rows);end=max(float(r['request_ended_at']) for r in rows)
 vals=[]
 with (OUT/'gpu1_telemetry.csv').open(newline='') as f:
  for r in csv.DictReader(f):
   if r['phase']!='q8_0:translation':continue
   t=__import__('datetime').datetime.fromisoformat(r['timestamp_utc']).timestamp()
   if start<=t<=end:vals.append(int(r['memory_used_mib']))
 if len(vals)<20:raise RuntimeError(f'Insufficient Q8 telemetry samples ({len(vals)})')
 peak=max(vals)
 return {'q8_observed_peak_mib':peak,'q8_observed_peak_gib':peak/1024,
  'q8_median_mib':statistics.median(vals),'q8_telemetry_samples':len(vals),
  'q8_sentences_per_second':meta['sentences_per_second'],'q8_generated_tokens_per_second':meta['generated_tokens_per_second'],
  'vllm_gpu_memory_utilization':peak/GPU_TOTAL_MIB}

def run_vllm(key,fraction,phase):
 ensure_gpu_idle();folder=OUT/key;folder.mkdir(exist_ok=True);phase.write_text(key+':startup')
 suffix={'bf16_vllm':'','fp8_vllm':'-FP8','nvfp4_vllm':'-FP4'}[key]
 cmd=[VLLM,'-m','vllm.entrypoints.openai.api_server','--model',str(MODELROOT/('Index-Translate-2B'+suffix)),
  '--served-model-name','Index-Translate-2B','--host','127.0.0.1','--port','8090','--dtype','bfloat16',
  '--max-model-len','1024','--max-num-seqs','128','--max-num-batched-tokens','8192',
  '--gpu-memory-utilization',f'{fraction:.6f}','--language-model-only','--no-enable-prefix-caching','--generation-config','vllm']
 if suffix:cmd += ['--quantization','compressed-tensors','--linear-backend','marlin']
 proc,logf=launch(cmd,folder/'server.log',ENV)
 save(folder/'server_config.json',{'command':cmd,'pid':proc.pid,'physical_gpu':1,
  'target_gpu_memory_utilization':fraction,'target_reference':'full-run Q8_0 measured nvidia-smi peak'})
 status='complete';error=None
 try:
  wait_health(proc,'http://127.0.0.1:8090',folder/'server.log')
  phase.write_text(key+':translation')
  run_client([BENCH,str(RUNS/'run_vllm_translation.py'),'--model-key',key,'--output',str(folder/'predictions.jsonl'),
   '--base-url','http://127.0.0.1:8090','--concurrency','128','--max-tokens','512','--timeout','600','--retries','4'],folder/'translation.log')
  m=json.loads((folder/'metadata.json').read_text())
  if m.get('status')!='complete' or m.get('rows_written')!=8096:raise RuntimeError('vLLM full run incomplete')
 except Exception as e:
  status='failed';error=str(e);save(folder/'status.json',{'status':status,'error':error})
 finally:
  stop_process_group(proc);logf.close();time.sleep(5)
 if status!='complete':return {'model':key,'status':status,'error':error}
 m=json.loads((folder/'metadata.json').read_text())
 old=load_rows(RUNS/key/'predictions.jsonl');new=load_rows(folder/'predictions.jsonl')
 ref={(r['language'],str(r['id'])):r['hypothesis'] for r in old}
 exact=sum(r['hypothesis']==ref[(r['language'],str(r['id']))] for r in new)
 memory=memory_during_requests(key,new)
 return {'model':key,'status':'complete','sentences_per_second':m['sentences_per_second'],
  'generated_tokens_per_second':m['generated_tokens_per_second'],'rows_written':m['rows_written'],
  'truncated':sum(bool(r.get('truncated')) for r in new),'gpu_memory_utilization':fraction,
  **memory,
  'exact_hypotheses_vs_high_vram_run':exact,'exact_hypothesis_fraction':exact/len(new),
  'dataset_sha256':m['dataset_sha256']}

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 if len(load_rows(QUANT/'q8_0/predictions.jsonl'))!=8096:raise RuntimeError('Existing Q8_0 full-run baseline missing/incomplete')
 phase,stop,collector=start_collector();summary={'status':'running','target':'full-run Q8_0 measured whole-GPU peak','models':[]}
 try:
  if (OUT/'q8_memory_budget.json').is_file():
   baseline=json.loads((OUT/'q8_memory_budget.json').read_text())
  else:
   run_q8(phase)
   baseline=q8_memory_budget();save(OUT/'q8_memory_budget.json',baseline)
  summary['q8_baseline']=baseline
  calibration_path=OUT/'calibration_summary.json'
  calibrated=json.loads(calibration_path.read_text())['models'] if calibration_path.is_file() else {}
  for key in VLLM_KEYS:
   util=baseline['vllm_gpu_memory_utilization']
   if key in calibrated:
    util=util*baseline['q8_observed_peak_mib']/calibrated[key]['peak_gpu_memory_mib']
   result=run_vllm(key,util,phase);summary['models'].append(result)
   save(OUT/'progress.json',summary)
  summary['status']='complete'
  save(OUT/'equal_vram_summary.json',summary)
 finally:
  stop.touch();collector.wait(timeout=12)

if __name__=='__main__':main()
