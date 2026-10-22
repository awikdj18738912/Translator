"""Protocol checks that do not load GPU models or application services."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('zh_bench', Path(__file__).resolve().parents[1] / 'scripts/benchmark_zh_translation.py')
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


class ProtocolTests(unittest.TestCase):
    def rows(self):
        return [{'id': i, 'language': lang, 'source': f'中文{i}', 'reference': f'reference {lang} {i}'}
                for lang in ('en', 'ja') for i in range(1, 11)]

    def test_same_seed_and_source_ids_for_every_direction(self):
        selected = bench.select_rows(self.rows(), ['en', 'ja'], 4, 42)
        self.assertEqual(selected, bench.select_rows(self.rows(), ['en', 'ja'], 4, 42))
        self.assertEqual([r['id'] for r in selected[:4]], [r['id'] for r in selected[4:]])
        self.assertEqual(len(bench.select_rows(self.rows(), ['en', 'ja'], 0, 42)), 20)

    def test_alignment_errors_are_not_silently_dropped(self):
        rows = self.rows()
        with self.assertRaises(ValueError):
            bench.select_rows(rows[:-1], ['en', 'ja'], 2, 42)
        rows[-1] = {**rows[-1], 'source': '另一句'}
        with self.assertRaises(ValueError):
            bench.select_rows(rows, ['en', 'ja'], 2, 42)

    def test_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / 'input.jsonl'
            row = self.rows()[0]
            file.write_text(json.dumps(row) + '\n' + json.dumps(row) + '\n')
            with self.assertRaises(ValueError):
                bench.read_jsonl(file)

    def test_author_specific_prompts(self):
        self.assertIn('翻译为英语', bench.prompt_for('index', '你好', 'en'))
        self.assertIn('只需要输出翻译后的结果', bench.prompt_for('hy', '你好', 'ja'))
        self.assertEqual(bench.prompt_for('lmt', '你好', 'en'),
                         'Translate the following text from Chinese into English:\nChinese: 你好\nEnglish:')

    def test_percentiles_interpolate(self):
        value = bench.distribution([1, 2, 3, 4])
        self.assertEqual(value['p50'], 2.5)
        self.assertAlmostEqual(value['p95'], 3.85)

    def test_all_launches_models_in_separate_processes(self):
        args = SimpleNamespace(model='all', model_root=Path('/models'), data=Path('/data.jsonl'),
            output=Path('/out'), gpu=1, languages='en,ja', limit=3, seed=42,
            max_new_tokens=128, warmup=2, cpu_threads=2, batch_size=16, resume=True)
        with patch.object(bench, 'validate_checkpoint'), patch.object(bench.subprocess, 'run') as child:
            bench.run(args)
        self.assertEqual(child.call_count, 3)
        self.assertEqual([c.args[0][4] for c in child.call_args_list], ['index', 'hy', 'lmt'])
        self.assertTrue(all('--resume' in c.args[0] for c in child.call_args_list))

    @unittest.skipUnless(importlib.util.find_spec('sacrebleu'), 'Install optional benchmark dependencies')
    def test_scoring_perfect_reference_and_truncation_count(self):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / 'predictions.jsonl'
            row = dict(id=1, language='en', source='这是一段测试文字。',
                reference='This is a sufficiently long test sentence.',
                hypothesis='This is a sufficiently long test sentence.',
                latency_seconds=2, ttft_seconds=.25, decode_seconds_after_first_token=1,
                generated_tokens_including_eos=10, truncated=True, session_id='test')
            file.write_text(json.dumps(row) + '\n')
            with contextlib.redirect_stdout(io.StringIO()):
                bench.score(SimpleNamespace(predictions=file, bleu_tokenizer='13a'))
            value = json.loads((file.parent / 'summary.json').read_text())['languages']['en']
            self.assertAlmostEqual(value['chrFpp'], 100)
            self.assertAlmostEqual(value['BLEU'], 100)
            self.assertEqual(value['truncated_outputs'], 1)
            self.assertEqual(value['generated_tokens_per_second_e2e'], 5)

    @unittest.skipUnless(importlib.util.find_spec('sacrebleu'), 'Install optional benchmark dependencies')
    def test_batch_timings_are_counted_once_for_throughput(self):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / 'predictions.jsonl'
            output = 'A sufficiently long English reference sentence.'
            rows = [dict(id=i, language='en', source='中文', reference=output, hypothesis=output,
                         batch_id='en:0', batch_size=2, batch_latency_seconds=4,
                         per_item_amortized_latency_seconds=2, batch_ttft_seconds=.5,
                         decode_seconds_after_first_token=3, generated_tokens_including_eos=9,
                         truncated=False, session_id='test') for i in (1, 2)]
            file.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            with contextlib.redirect_stdout(io.StringIO()):
                bench.score(SimpleNamespace(predictions=file, bleu_tokenizer='13a'))
            value = json.loads((file.parent / 'summary.json').read_text())['languages']['en']
            self.assertEqual(value['batch_count'], 1)
            self.assertEqual(value['batch_latency_seconds']['mean'], 4)
            self.assertEqual(value['per_item_amortized_latency_seconds']['mean'], 2)
            self.assertEqual(value['sentences_per_second'], .5)
            self.assertEqual(value['max_batch_size'], 2)

    def fixture(self, root, name):
        folder = root / name
        folder.mkdir()
        row = dict(id=1, language='en', source='你好。', reference='This is a long test sentence.',
            hypothesis='This is a long test sentence.', latency_seconds=1,
            batch_id='en:0', batch_size=1, batch_latency_seconds=1,
            batch_ttft_seconds=.1,
            ttft_seconds=.1, decode_seconds_after_first_token=.9,
            generated_tokens_including_eos=9, truncated=False, session_id='test')
        file = folder / 'predictions.jsonl'
        file.write_text(json.dumps(row) + '\n')
        with contextlib.redirect_stdout(io.StringIO()):
            bench.score(SimpleNamespace(predictions=file, bleu_tokenizer='13a'))
        protocol = {k: 'same' for k in ('protocol', 'data_sha256', 'precision', 'batch_size',
            'max_new_tokens', 'decode', 'attention', 'torch', 'transformers', 'gpu',
            'device', 'seed', 'cpu_threads', 'warmup_per_language', 'script_sha256')}
        protocol['case_ids'] = [['en', 1]]
        bench.save_json(folder / 'metadata.json', {'status': 'complete', 'protocol': protocol})
        return folder

    @unittest.skipUnless(importlib.util.find_spec('sacrebleu'), 'Install optional benchmark dependencies')
    def test_report_checks_coverage_and_metric_signatures(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = self.fixture(root, 'index')
            second = self.fixture(root, 'hy')
            args = SimpleNamespace(output=root, models='index,hy')
            with contextlib.redirect_stdout(io.StringIO()):
                bench.report(args)
            self.assertTrue((root / 'comparison.md').is_file())
            summary = json.loads((second / 'summary.json').read_text())
            summary['languages']['en']['bleu_signature'] = 'different tokenizer'
            bench.save_json(second / 'summary.json', summary)
            with self.assertRaisesRegex(ValueError, 'Metric signatures'):
                bench.report(args)
            metadata = json.loads((first / 'metadata.json').read_text())
            metadata['protocol']['case_ids'].append(['en', 2])
            bench.save_json(first / 'metadata.json', metadata)
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                bench.report(args)


if __name__ == '__main__':
    unittest.main()
