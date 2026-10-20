"""Validate and merge the two completed LMT-60-0.6B evaluation shards."""
import json
from pathlib import Path

ROOT = Path('results/lmt600m_comparison_20261017')
MODEL = 'LMT-60-0.6B'
LANGUAGE_ORDER = ('en', 'fr', 'de', 'es', 'ru', 'ja', 'ko', 'ar')
SHARD_LANGUAGES = {
    'shard0': ('en', 'fr', 'de', 'es'),
    'shard1': ('ru', 'ja', 'ko', 'ar'),
}
MATCH_KEYS = ('revision', 'data_sha256', 'compute_type', 'beam_size',
              'max_decoding_length', 'runtime', 'ctranslate2',
              'transformers', 'sacrebleu')


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    shards = {}
    for shard, languages in SHARD_LANGUAGES.items():
        directory = ROOT / shard
        paths = {
            'summary': directory / f'{MODEL}.summary.json',
            'metadata': directory / f'{MODEL}.metadata.json',
            'offline': directory / f'{MODEL}.offline.jsonl',
            'streaming': directory / f'{MODEL}.streaming.jsonl',
            'protocol_audit': directory / f'{MODEL}.protocol_audit.json',
        }
        missing = [str(path) for path in paths.values() if not path.exists()]
        require(not missing, f'{shard}: missing expected artifacts: {missing}')
        shards[shard] = {
            'languages': languages,
            **{key: json.loads(path.read_text()) for key, path in paths.items()
               if key in ('summary', 'metadata', 'protocol_audit')},
            'offline': read_jsonl(paths['offline']),
            'streaming': read_jsonl(paths['streaming']),
        }

    base = shards['shard0']['metadata']
    for shard, result in shards.items():
        meta, summary = result['metadata'], result['summary']
        require(summary.get('status') == 'completed', f'{shard}: summary is not completed')
        require(meta.get('status', 'completed') == 'completed', f'{shard}: metadata is not completed')
        require(meta.get('offline_batch_size') == 1, f'{shard}: offline batch size is not 1')
        for key in MATCH_KEYS:
            require(meta.get(key) == base.get(key), f'{shard}: metadata mismatch for {key}')
            require(summary.get(key) == base.get(key), f'{shard}: summary mismatch for {key}')
        require(summary.get('offline_batch_size') == 1, f'{shard}: summary offline batch size is not 1')
        require(summary.get('offline_subset_only') is False, f'{shard}: offline data is not full input')

        expected_languages = set(result['languages'])
        offline_languages = {row['language'] for row in result['offline']}
        stream_languages = {row['language'] for row in result['streaming']}
        audit_languages = [row['language'] for row in result['protocol_audit']]
        require(offline_languages == expected_languages, f'{shard}: offline languages differ from shard assignment')
        require(stream_languages == expected_languages, f'{shard}: streaming languages differ from shard assignment')
        require(set(audit_languages) == expected_languages and len(audit_languages) == len(expected_languages),
                f'{shard}: protocol audit languages differ from shard assignment')
        require(len(result['offline']) == len(expected_languages) * 1012,
                f'{shard}: unexpected offline row count')
        require(len(result['streaming']) == len(expected_languages) * 100 * 2,
                f'{shard}: unexpected streaming trajectory count')
        require(set(summary.get('scores', {})) == expected_languages,
                f'{shard}: summary score languages differ from shard assignment')
        for lang in result['languages']:
            require(sum(row['language'] == lang for row in result['offline']) == 1012,
                    f'{shard}/{lang}: expected 1012 offline rows')
            for mode in ('retranslate', 'agreement_forced_prefix'):
                require(sum(row['language'] == lang and row['mode'] == mode
                            for row in result['streaming']) == 100,
                        f'{shard}/{lang}/{mode}: expected 100 stream trajectories')

    offline = [row for shard in SHARD_LANGUAGES for row in shards[shard]['offline']]
    streaming = [row for shard in SHARD_LANGUAGES for row in shards[shard]['streaming']]
    audit_rows = [row for shard in SHARD_LANGUAGES for row in shards[shard]['protocol_audit']]
    require(len(offline) == 8096, 'merged offline row count is not 8096')
    require(len({(row['language'], row['id']) for row in offline}) == 8096,
            'merged offline IDs are not unique')
    require(len(streaming) == 1600, 'merged streaming trajectory count is not 1600')
    require(len({(row['language'], row['id'], row['mode']) for row in streaming}) == 1600,
            'merged streaming trajectory IDs are not unique')
    require(set(row['language'] for row in offline) == set(LANGUAGE_ORDER),
            'merged offline languages do not cover all eight languages')
    require(set(row['language'] for row in streaming) == set(LANGUAGE_ORDER),
            'merged streaming languages do not cover all eight languages')
    require(set(row['language'] for row in audit_rows) == set(LANGUAGE_ORDER)
            and len(audit_rows) == 8, 'merged protocol audit does not cover all eight languages')

    # Impose the benchmark's canonical language order while preserving file order
    # within each language.
    language_rank = {language: index for index, language in enumerate(LANGUAGE_ORDER)}
    offline.sort(key=lambda row: language_rank[row['language']])
    streaming.sort(key=lambda row: language_rank[row['language']])
    audit_rows.sort(key=lambda row: language_rank[row['language']])

    scores = {}
    for language in LANGUAGE_ORDER:
        source_shard = next(shard for shard, result in shards.items()
                            if language in result['languages'])
        scores[language] = shards[source_shard]['summary']['scores'][language]

    shard_times = {shard: shards[shard]['summary']['total_seconds'] for shard in SHARD_LANGUAGES}
    require(all(isinstance(seconds, (int, float)) and seconds >= 0
                for seconds in shard_times.values()), 'invalid shard total_seconds')
    worker_seconds = sum(shard_times.values())
    merged_summary = dict(shards['shard0']['summary'])
    merged_summary.update({
        'gpu_index': None,
        'gpu_indices': [0, 1],
        'gpu_shards': ['shard0', 'shard1'],
        'offline_batch_size': 1,
        'offline_batch_size_source': 'Both shard metadata files record offline_batch_size=1.',
        'status': 'completed',
        'completed_languages': list(LANGUAGE_ORDER),
        'offline_pairs': len(offline),
        'streaming_trajectories': len(streaming),
        'scores': scores,
        'shard_times_seconds': shard_times,
        'sum_gpu_worker_seconds': worker_seconds,
        'total_seconds': max(shard_times.values()),
        'total_seconds_semantics': 'Maximum shard worker duration; concurrent GPU worker time is not end-to-end wall-clock time.',
        'shard_gpu_snapshots': {
            shard: {'start': shards[shard]['summary'].get('gpu_snapshot_start'),
                    'end': shards[shard]['summary'].get('gpu_snapshot_end')}
            for shard in SHARD_LANGUAGES
        },
    })
    merged_summary.pop('gpu_snapshot_start', None)
    merged_summary.pop('gpu_snapshot_end', None)

    merged_metadata = dict(shards['shard0']['metadata'])
    merged_metadata.pop('gpu_index', None)
    merged_metadata.update({
        'gpu_indices': [0, 1],
        'gpu_shards': ['shard0', 'shard1'],
        'offline_batch_size': 1,
        'offline_batch_size_source': 'Both shard metadata files record offline_batch_size=1.',
        'shard_metadata': {shard: shards[shard]['metadata'] for shard in SHARD_LANGUAGES},
        'shard_times_seconds': shard_times,
        'sum_gpu_worker_seconds': worker_seconds,
        'total_seconds': max(shard_times.values()),
        'total_seconds_semantics': 'Maximum shard worker duration; concurrent GPU worker time is not end-to-end wall-clock time.',
    })
    merged_progress = dict(merged_metadata)
    merged_progress.update({
        'status': 'completed',
        'completed_languages': list(LANGUAGE_ORDER),
        'offline_pairs': len(offline),
        'streaming_trajectories': len(streaming),
    })

    prefix = ROOT / MODEL
    write_jsonl(ROOT / f'{MODEL}.offline.jsonl', offline)
    write_jsonl(ROOT / f'{MODEL}.streaming.jsonl', streaming)
    (ROOT / f'{MODEL}.summary.json').write_text(json.dumps(merged_summary, ensure_ascii=False, indent=2) + '\n')
    (ROOT / f'{MODEL}.metadata.json').write_text(json.dumps(merged_metadata, ensure_ascii=False, indent=2) + '\n')
    (ROOT / f'{MODEL}.progress.json').write_text(json.dumps(merged_progress, ensure_ascii=False, indent=2) + '\n')
    (ROOT / f'{MODEL}.protocol_audit.json').write_text(json.dumps(audit_rows, ensure_ascii=False, indent=2) + '\n')
    print(prefix)


if __name__ == '__main__':
    main()
