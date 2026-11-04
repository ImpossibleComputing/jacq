import os
from pathlib import Path

path = Path(os.environ['VLLM_ROOT']).resolve() / 'vllm/v1/attention/backends/kivi_dsl.py'
source = path.read_text()
assert 'def _trace_op(' not in source
source = source.replace(
    'import torch\n',
    '''import torch


def _trace_op(name: str) -> None:
    import os
    path = os.environ.get("JACQ_TRACE_OPS")
    if path:
        with open(path, "a", encoding="utf-8") as trace:
            trace.write(f"{os.getpid()} {name}\\n")
''',
    1,
)
for call, name, expected in [
    ('torch.ops._C.kivi_decode_dsl(*args)', 'kivi_decode_dsl', 2),
    ('torch.ops._C.kivi_decode_sparse_retained_dsl(*args)', 'kivi_decode_sparse_retained_dsl', 2),
    ('torch.ops._C.kivi_decode_batched_dsl(*args)', 'kivi_decode_batched_dsl', 2),
    ('op(*tensors, seq_len)', 'cache_write:' + '{op_name}', 1),
    ('torch.ops._C.kivi_cache_write_batched_dsl(*tensors, start_seq_len, num_new)', 'kivi_cache_write_batched_dsl', 1),
]:
    assert source.count(call) == expected, (call, source.count(call))
    label = 'f"cache_write:{op_name}"' if name.startswith('cache_write:') else repr(name)
    source = source.replace(call, f'{call}; _trace_op({label})')
path.write_text(source)
