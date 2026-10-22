import pytest
from system.qwen_asr_stream_server import parse_args,vllm_scheduling_options


def test_serial_profile_budget_is_forwarded(tmp_path):
    args=parse_args(['--model',str(tmp_path),'--max-num-seqs','1','--max-num-batched-tokens','4096'])
    assert vllm_scheduling_options(args)=={'max_num_seqs':1,'max_num_batched_tokens':4096}


def test_unset_profile_budget_keeps_library_defaults(tmp_path):
    assert vllm_scheduling_options(parse_args(['--model',str(tmp_path)]))=={}


def test_explicit_cache_budget_is_forwarded(tmp_path):
    args=parse_args(['--model',str(tmp_path),'--kv-cache-memory-bytes','1073741824'])
    assert vllm_scheduling_options(args)=={'kv_cache_memory_bytes':1073741824}


def test_eager_execution_is_explicit_and_forwarded(tmp_path):
    args=parse_args(['--model',str(tmp_path),'--enforce-eager'])
    assert vllm_scheduling_options(args)=={'enforce_eager':True}


@pytest.mark.parametrize('value',['0','-1'])
def test_invalid_cache_budget_rejected(tmp_path,value):
    with pytest.raises(SystemExit):
        parse_args(['--model',str(tmp_path),'--kv-cache-memory-bytes',value])


@pytest.mark.parametrize('flags',[['--max-num-seqs','0'],['--max-num-batched-tokens','0'],['--max-num-seqs','5','--max-num-batched-tokens','4']])
def test_invalid_profile_budget_rejected(tmp_path,flags):
    with pytest.raises(SystemExit):parse_args(['--model',str(tmp_path),*flags])
