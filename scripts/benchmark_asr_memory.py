#!/usr/bin/env python3
"""Compare live streaming ASR memory profiles on GPU0, retaining raw evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time
import wave

import av
import numpy as np
import pynvml
import requests

ROOT = Path(__file__).resolve().parents[1]
UNIT = 'translation-workbench-asr.service'
URL = 'http://127.0.0.1:8766'
RATE = 16000
MIB = 1024 ** 2
CONFIGS = [
    dict(label='baseline_8192', context=8192, kv_mib=1024, prefill=4096, eager=False),
    dict(label='compact_4096', context=4096, kv_mib=512, prefill=1024, eager=False),
    dict(label='compact_2048', context=2048, kv_mib=256, prefill=1024, eager=False),
    dict(label='eager_4096', context=4096, kv_mib=512, prefill=1024, eager=True),
]


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def shell(args, **kwargs):
    return subprocess.run(args, capture_output=True, text=True, check=True, **kwargs).stdout.strip()


def decode(path):
    frames = []
    with av.open(str(path)) as container:
        converter = av.AudioResampler(format='fltp', layout='mono', rate=RATE)
        for frame in container.decode(audio=0):
            frames.extend(x.to_ndarray().reshape(-1) for x in converter.resample(frame))
        frames.extend(x.to_ndarray().reshape(-1) for x in converter.resample(None))
    return np.concatenate(frames).astype('<f4')


def descendants(pid):
    ids = {pid}
    parents = {}
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            try:
                parents[int(entry.name)] = int((entry / 'stat').read_text().rsplit(')', 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                pass
    while True:
        expanded = ids | {child for child, parent in parents.items() if parent in ids}
        if ids == expanded:
            return ids
        ids = expanded


class Memory:
    def __init__(self):
        self.main_pid = int(shell(['systemctl', '--user', 'show', UNIT, '-p', 'MainPID', '--value']))
        self.pids = descendants(self.main_pid)
        self.handle = pynvml.nvmlDeviceGetHandleByUUID('GPU-c3da2483-42aa-6594-0038-5deef8baef11')
        self.phase = 'idle'
        self.rows = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def sample(self):
        try:
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
            used = {int(p.pid): int(p.usedGpuMemory) for p in processes
                    if p.pid in self.pids and p.usedGpuMemory < 2**60}
            self.rows.append(dict(time=time.time(), phase=self.phase,
                                  process_mib=sum(used.values()) / MIB,
                                  device_mib=pynvml.nvmlDeviceGetMemoryInfo(self.handle).used / MIB,
                                  pids=used))
        except Exception as exc:
            self.rows.append(dict(time=time.time(), phase=self.phase, error=repr(exc)))

    def loop(self):
        while not self.stop_event.is_set():
            self.sample()
            self.stop_event.wait(0.2)

    def __enter__(self):
        self.sample()
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join()
        self.sample()

    def summary(self):
        valid = [x for x in self.rows if 'process_mib' in x]
        return dict(main_pid=self.main_pid, pids=sorted(self.pids),
                    idle_initial_mib=valid[0]['process_mib'] if valid else None,
                    peak_process_mib=max((x['process_mib'] for x in valid), default=None),
                    peak_device_mib=max((x['device_mib'] for x in valid), default=None),
                    idle_after_mib=valid[-1]['process_mib'] if valid else None,
                    sample_interval_seconds=0.2, samples=self.rows)


def wait_ready(session, timeout=540):
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        state = shell(['systemctl', '--user', 'show', UNIT, '-p', 'ActiveState', '--value'])
        if state in ('failed', 'inactive'):
            raise RuntimeError(f'ASR service is {state}')
        try:
            result = session.get(URL + '/health', timeout=2)
            if result.status_code == 200 and result.json().get('ok'):
                return result.json()
        except requests.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError('ASR health did not become ready within 540 seconds')


def run_audio(session, name, samples, language, *, context=''):
    record = dict(name=name, audio_seconds=len(samples)/RATE, language=language,
                  context_chars=len(context), chunks=[], errors=[], http_seconds=0)
    sid = None

    def post(endpoint, **kwargs):
        started = time.perf_counter()
        try:
            response = session.post(URL + endpoint, timeout=(5, 180), **kwargs)
            response.raise_for_status()
            payload = response.json()
            if payload.get('error'):
                raise RuntimeError(str(payload['error']))
            return payload
        finally:
            record['http_seconds'] += time.perf_counter() - started

    try:
        opened = post('/stream/start', params=dict(language=language, chunk_seconds=2, context=context))
        sid = opened['session_id']
        for offset in range(0, len(samples), 2 * RATE):
            started = time.perf_counter()
            payload = post('/stream/chunk', params=dict(session_id=sid),
                           data=samples[offset:offset+2*RATE].astype('<f4').tobytes(),
                           headers={'Content-Type': 'application/octet-stream'})
            record['chunks'].append(dict(offset_seconds=offset/RATE,
                                         seconds=time.perf_counter()-started, payload=payload))
        record['final'] = post('/stream/finish', params=dict(session_id=sid))
        record['text'] = record['final'].get('text', '')
        sid = None
    except Exception as exc:
        record['errors'].append(repr(exc))
    finally:
        if sid:
            try:
                session.post(URL + '/stream/cancel', params=dict(session_id=sid), timeout=10)
            except requests.RequestException:
                pass
    record['rtf_http'] = record['http_seconds'] / record['audio_seconds']
    values = [x['seconds'] for x in record['chunks']]
    record['chunk_p95_seconds'] = float(np.percentile(values, 95)) if values else None
    return record


def fixtures(output):
    paths = dict(english=ROOT/'workbench/tests/fixtures/english_probe.wav',
                 chinese_short=ROOT/'results/chinese_first_pipeline_20261020/chinese_probe.wav',
                 chinese_real=ROOT.parent/'Codex-web/测试音频1.mp3')
    audio = {name: decode(path) for name, path in paths.items()}
    base = audio['chinese_real']
    audio['chinese_repeated_240s'] = np.tile(base, int(np.ceil(240*RATE/len(base))))[:240*RATE]
    frames = base[:len(base)//320*320].reshape(-1, 320)
    dense = frames[np.sqrt(np.mean(frames**2, axis=1)) > .008].reshape(-1)
    audio['dense_speech_60s'] = np.tile(dense, int(np.ceil(60*RATE/len(dense))))[:60*RATE]
    dense_path = output/'dense_speech_60s.wav'
    with wave.open(str(dense_path), 'wb') as target:
        target.setparams((1, 2, RATE, 0, 'NONE', 'not compressed'))
        target.writeframes((np.clip(audio['dense_speech_60s'], -1, 1)*32767).astype('<i2').tobytes())
    audio['dense_speech_60s'] = decode(dense_path)
    metadata = {}
    for name, pcm in audio.items():
        metadata[name] = dict(duration_seconds=len(pcm)/RATE,
                              pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest())
        if name in paths:
            metadata[name].update(path=str(paths[name]),
                                  file_sha256=hashlib.sha256(paths[name].read_bytes()).hexdigest())
    metadata['chinese_repeated_240s']['description'] = 'Real 40.848s Chinese recording repeated to 240s; not natural continuous speech'
    metadata['dense_speech_60s']['description'] = 'Artificial capacity stress: concatenate 20ms frames with RMS > 0.008 and repeat to 60s'
    save(output/'fixtures.json', metadata)
    return audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--labels', nargs='+', choices=[x['label'] for x in CONFIGS])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    override = Path.home()/'.config/systemd/user'/f'{UNIT}.d/90-memory-experiment.conf'
    if override.exists():
        raise SystemExit(f'Refusing to overwrite an existing experiment: {override}')
    override.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False
    pynvml.nvmlInit()
    audio = fixtures(args.output)
    progress = dict(state='running', completed=[], current=None)
    successful = False
    try:
        configs = [x for x in CONFIGS if not args.labels or x['label'] in args.labels]
        for config in configs:
            label = config['label']
            progress['current'] = label
            progress['phase'] = 'startup'
            save(args.output/'progress.json', progress)
            started_at = datetime.now(timezone.utc).isoformat()
            started_epoch = int(time.time())
            print(json.dumps(dict(event='start', **config)), flush=True)
            result = dict(config=config, started_at=started_at, runs=[], warmups=[], capacity=[])
            try:
                try:
                    current = session.get(URL + '/health', timeout=2).json()
                except (requests.RequestException, ValueError):
                    current = {}
                matches = (current.get('max_model_len') == config['context']
                           and current.get('kv_cache_memory_bytes') == config['kv_mib']*MIB
                           and current.get('max_num_batched_tokens', 4096) == config['prefill']
                           and current.get('enforce_eager', False) == config['eager'])
                if not matches:
                    # All benchmark sessions are drained before changing the service.
                    # Old vLLM workers can linger after TERM; bound that cleanup.
                    override.write_text('[Service]\nTimeoutStopSec=20\n' + '\n'.join([
                        'Environment=ASR_GPU=0',
                        f"Environment=ASR_MAX_MODEL_LEN={config['context']}",
                        f"Environment=ASR_KV_CACHE_BYTES={config['kv_mib']*MIB}",
                        f"Environment=ASR_MAX_BATCHED_TOKENS={config['prefill']}",
                        f"Environment=ASR_ENFORCE_EAGER={int(config['eager'])}",
                    ]) + '\n')
                    shell(['systemctl', '--user', 'daemon-reload'])
                    shell(['systemctl', '--user', 'restart', UNIT], timeout=120)
                result['health'] = wait_ready(session)
                result['command'] = shell(['systemctl', '--user', 'show', UNIT, '-p', 'ExecStart', '--value'])
                with Memory() as mem:
                    for name in ('english', 'chinese_repeated_240s'):
                        mem.phase = 'warmup:' + name
                        progress['phase'] = mem.phase
                        save(args.output/'progress.json', progress)
                        record = run_audio(session, name, audio[name], 'English' if name == 'english' else 'Chinese')
                        result['warmups'].append(record)
                        if record['errors']:
                            raise RuntimeError(f"Warmup {name} failed: {record['errors']}")
                    for repeat in range(args.repeats):
                        for name, pcm in audio.items():
                            mem.phase = f'measure:{repeat}:{name}'
                            progress['phase'] = mem.phase
                            save(args.output/'progress.json', progress)
                            record = run_audio(session, name, pcm, 'English' if name == 'english' else 'Chinese')
                            record['repeat'] = repeat
                            result['runs'].append(record)
                            print(json.dumps(dict(event='case', label=label, name=name, repeat=repeat,
                                                  seconds=round(record['http_seconds'], 3), errors=record['errors'])), flush=True)
                    mem.phase = 'capacity:max_hint'
                    record = run_audio(session, 'max_hint', audio['chinese_short'], 'Chinese',
                                       context=('甲乙丙丁戊己庚辛壬癸'*205)[:2048])
                    result['capacity'].append(record)
                    mem.phase = 'idle_after'
                    mem.sample()
                result['gpu_memory'] = mem.summary()
            except Exception as exc:
                result['failure'] = repr(exc)
            result['journal'] = shell(['journalctl', '--user', '-u', UNIT, '--since', f'@{started_epoch}',
                                       '--no-pager', '-o', 'cat'])
            save(args.output/f'{label}.json', result)
            progress['completed'].append(label)
            print(json.dumps(dict(event='complete', label=label, failure=result.get('failure'),
                                  peak_mib=result.get('gpu_memory', {}).get('peak_process_mib'))), flush=True)
        progress.update(state='awaiting_selection', phase='complete')
        save(args.output/'progress.json', progress)
        successful = True
    finally:
        if not successful and override.exists():
            shell(['systemctl', '--user', 'stop', UNIT], timeout=120)
            override.unlink()
            shell(['systemctl', '--user', 'daemon-reload'])
            shell(['systemctl', '--user', 'start', UNIT], timeout=120)
            wait_ready(session)
        pynvml.nvmlShutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
