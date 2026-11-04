#!/usr/bin/env python3
"""Run three isolated, sequential vLLM serving arms on one local GPU."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

VLLM = Path(os.environ['VLLM_ROOT']).resolve()
EVIDENCE = Path(os.environ.get('JACQ_TEST_EVIDENCE_DIR', 'evidence')).resolve()
MAX_MODEL_LEN = int(os.environ.get('JACQ_TEST_MAX_MODEL_LEN', '4096'))
WAVES = int(os.environ.get('JACQ_TEST_WAVES', '12'))
MODEL = Path(os.environ['MODEL_PATH']).resolve()
PYTHON = os.environ.get('PYTHON_EXECUTABLE', sys.executable)
PORT = int(os.environ.get('JACQ_TEST_PORT', '18947'))
BASE = f'http://127.0.0.1:{PORT}'


def api(method, route, body=None, timeout=15):
    payload = None if body is None else json.dumps(body).encode()
    request = Request(BASE + route, payload, {'Content-Type': 'application/json'}, method=method)
    with urlopen(request, timeout=timeout) as response:
        return json.load(response) if route != '/health' else response.status


def requests_for_arm():
    short = ('Read this list carefully: ' + ' '.join(f'item{i}=blue' for i in range(18))
             + ' Answer with one word. The sky is')
    long = ('The secret code is LIME42. ' + ' '.join(f'entry{i}=amber' for i in range(42))
            + ' Question: What is the secret code? Answer:')
    cases = [('short_one', short, 1), ('long_one', long, 1),
             ('needle', long, 8)]
    cases += [(f'wave_{i:02d}', long if i % 2 else short, 2) for i in range(WAVES)]
    return cases


def run_arm(name, cache_dtype, writer):
    env = os.environ.copy()
    env.update({
        'PYTHONPATH': str(VLLM),
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', '1'),
        'VLLM_USE_V1': '1',
        'JAC_KIVI_BATCHED_PREFILL': '0',
        'JAC_KIVI_SPARSE_RETAINED_DECODE': '0',
        'JAC_KIVI_CACHE_WRITE_KERNEL': writer,
        'JACQ_TRACE_OPS': str(EVIDENCE / f'{name}.ops.log'),
        'PYTHONUNBUFFERED': '1',
        'HF_HUB_OFFLINE': '1',
        'TRANSFORMERS_OFFLINE': '1',
    })
    command = [PYTHON, '-m', 'vllm.entrypoints.openai.api_server',
               '--host', '127.0.0.1', '--port', str(PORT),
               '--model', str(MODEL), '--served-model-name', 'jacq-14b',
               '--kv-cache-dtype', cache_dtype, '--block-size', '32',
               '--max-model-len', str(MAX_MODEL_LEN), '--max-num-batched-tokens', '512',
               '--max-num-seqs', '2', '--tensor-parallel-size', '1',
               '--gpu-memory-utilization', '0.7', '--enforce-eager',
               '--no-enable-prefix-caching', '--disable-log-requests']
    result = {'arm': name, 'cache_dtype': cache_dtype, 'writer': writer,
              'command': command, 'env_overrides': {k: env[k] for k in
              ('PYTHONPATH', 'CUDA_VISIBLE_DEVICES', 'VLLM_USE_V1',
               'JAC_KIVI_BATCHED_PREFILL', 'JAC_KIVI_SPARSE_RETAINED_DECODE',
               'JAC_KIVI_CACHE_WRITE_KERNEL', 'JACQ_TRACE_OPS', 'HF_HUB_OFFLINE')},
              'requests': []}
    log_path = EVIDENCE / f'{name}.server.log'
    print(f'Starting {name}', flush=True)
    with log_path.open('w') as server_log:
        process = subprocess.Popen(command, env=env, cwd=VLLM, stdout=server_log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        result['server_pid'] = process.pid
        try:
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f'server exited {process.returncode} before ready')
                try:
                    if api('GET', '/health', timeout=2) == 200:
                        break
                except (URLError, HTTPError, TimeoutError):
                    pass
                time.sleep(2)
            else:
                raise TimeoutError('server did not become healthy in 300 seconds')
            result['models'] = api('GET', '/v1/models')
            for case, prompt, max_tokens in requests_for_arm():
                body = {'model': 'jacq-14b', 'prompt': prompt,
                        'temperature': 0, 'max_tokens': max_tokens, 'seed': 42}
                t0 = time.monotonic()
                try:
                    response = api('POST', '/v1/completions', body, timeout=120)
                    result['requests'].append({'case': case, 'prompt': prompt,
                                               'max_tokens': max_tokens,
                                               'elapsed_s': round(time.monotonic()-t0, 3),
                                               'response': response})
                    print(name, case, response['choices'][0]['text'], response['usage'], flush=True)
                except Exception as exc:
                    result['requests'].append({'case': case, 'error': repr(exc)})
                    raise
            chat_body = {'model': 'jacq-14b', 'messages': [
                {'role': 'user', 'content': 'Reply with the single word READY.'}],
                'temperature': 0, 'max_tokens': 16, 'seed': 42}
            result['chat_response'] = api('POST', '/v1/chat/completions', chat_body, timeout=120)
            print(name, 'chat', result['chat_response']['choices'][0]['message']['content'], flush=True)
            result['status'] = 'passed_requests'
        except Exception as exc:
            result['status'] = 'failed'
            result['error'] = repr(exc)
            print(name, 'FAILED', repr(exc), flush=True)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            result['server_exit'] = process.returncode
            result['op_counts'] = {}
            trace_path = EVIDENCE / f'{name}.ops.log'
            if trace_path.exists():
                for line in trace_path.read_text().splitlines():
                    op = line.split(' ', 1)[1]
                    result['op_counts'][op] = result['op_counts'].get(op, 0) + 1
            (EVIDENCE / f'{name}.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    EVIDENCE.mkdir(exist_ok=True)
    arms = [
        ('baseline_fp16', 'auto', 'hand_cuda'),
        ('kivi_hand_cuda', 'int4_kivi', 'hand_cuda'),
        ('kivi_jac_dsl', 'int4_kivi', 'jac_dsl'),
    ]
    selected = os.environ.get('JACQ_TEST_ARMS')
    if selected:
        names = set(selected.split(','))
        arms = [arm for arm in arms if arm[0] in names]
    for name, dtype, writer in arms:
        result = run_arm(name, dtype, writer)
        if result['status'] != 'passed_requests':
            return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
