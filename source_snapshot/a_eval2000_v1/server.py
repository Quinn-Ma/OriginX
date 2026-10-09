"""Explicit parity or localhost service wrapper. Never modifies existing endpoints."""
import argparse
import json
import os
from pathlib import Path
import secrets
import signal
import threading
import time
from .loader import load_bound_model, require, sha


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def check_lora_hello(actual, expected):
    """Existing client checks only the base; callers MUST additionally call this."""
    require(actual.get('lora_serving_identity') == expected.get('lora_serving_identity')
            and bool(expected.get('lora_serving_identity')), 'LoRA endpoint identity differs')
    require(actual.get('loader_kind') == expected.get('loader_kind'), 'Wrong model loader')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('parity','serve'),required=True)
    parser.add_argument('--binding',type=Path,required=True)
    parser.add_argument('--binding-sha256',required=True)
    parser.add_argument('--port',type=int,default=0)
    parser.add_argument('--max-clients',type=int,default=6)
    parser.add_argument('--request-a',type=Path)
    parser.add_argument('--request-b',type=Path)
    parser.add_argument('--report',type=Path)
    parser.add_argument('--parity',type=Path)
    parser.add_argument('--parity-sha256')
    parser.add_argument('--server-manifest',type=Path)
    args = parser.parse_args()
    require(1 <= args.max_clients <= 8, 'Existing multiplex capacity is1..8')
    if args.mode == 'parity':
        require(args.request_a and args.request_b and args.report and not args.report.exists(), 'Need two real processor fixtures and new report')
    else:
        require(args.parity and args.parity_sha256 and args.server_manifest
                and not args.server_manifest.exists() and 1024 <= args.port <= 65535,
                'Need actual passed parity, new manifest and dedicated fixed port')
    # Explicit deterministic backend settings; apply equally to direct/socket.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    import torch
    from multiplex_inference.core import SerialExecutor
    from multiplex_inference.server import serve
    import multiplex_inference.server as transport
    require(torch.cuda.is_available() and torch.cuda.device_count()==1, 'Explicitly reserve one healthy visible GPU')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    model, handle, identity, binding = load_bound_model(args.binding,args.binding_sha256)
    sources = {p.name:sha(p) for p in sorted(Path(transport.__file__).parent.glob('*.py'))}
    identity.update(server_instance=secrets.token_hex(16),multiplex_sources_sha256=sources,
                    torch_version=torch.__version__,cuda_visible_device_count=1)
    if args.mode == 'parity':
        from multiplex_inference.parity import load_request_fixture,check_sequences
        requests=[load_request_fixture(p,torch) for p in (args.request_a,args.request_b)]
        result=check_sequences(model,requests,torch=torch,identity=identity,port=args.port)
        result.update(model=identity,source_sha256=sources,
                      fixture_sha256={str(p):sha(p) for p in (args.request_a,args.request_b)},
                      fixture_npz_sha256={str(p.with_suffix('.npz')):sha(p.with_suffix('.npz')) for p in (args.request_a,args.request_b)},
                      note='Same nonzero adapter direct/socket inference parity; not training-vs-inference precision equivalence or rollout benefit.')
        write_new(args.report,result)
        require(result['passed'],'Same-adapter socket parity failed')
        return
    require(sha(args.parity)==args.parity_sha256,'Parity artifact changed')
    parity=json.loads(args.parity.read_text())
    require(parity.get('passed') is True and len(parity.get('trace',[]))==6
            and all(x['full_action_exact'] and x['all_rng_exact'] for x in parity['trace'])
            and parity.get('ambient_rng_unchanged') is True
            and parity['source_sha256']==sources
            and parity['model']['model_assets_sha256']==identity['model_assets_sha256']
            and parity['model']['torch_version']==identity['torch_version'], 'Parity does not cover this runtime/base')
    check_lora_hello(identity,parity['model'])
    executor=SerialExecutor(model,torch,identity,capacity=args.max_clients*2)
    stop=threading.Event();ready=threading.Event();info={};write_errors=[]
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.set())
    def record_ready():
        while not ready.wait(.2):
            if stop.is_set():return
        try:
            owner={'pid':os.getpid(),'process_start_ticks':int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
                   'command':Path('/proc/self/cmdline').read_bytes().decode().rstrip('\0').split('\0'),
                   'port':info['port'],'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')}
            write_new(args.server_manifest,{'ready':True,'repo':os.getcwd(),'model':binding['base_model'],
                     'source_sha256':sources,'hello':executor.identity,'servers':[owner],
                     'max_clients':args.max_clients,'parity_report':str(args.parity.resolve()),
                     'parity_sha256':args.parity_sha256,'lora_binding':str(args.binding.resolve()),
                     'lora_binding_sha256':args.binding_sha256,'created_unix':time.time()})
        except Exception as error:
            write_errors.append(error);stop.set()
    observer=threading.Thread(target=record_ready,daemon=True);observer.start()
    try:
        serve(executor,host='127.0.0.1',port=args.port,max_clients=args.max_clients,
              stop=stop,ready_identity=executor.identity,ready_event=ready,ready_info=info)
    finally:
        stop.set();observer.join(timeout=2);executor.close()
    if write_errors:raise RuntimeError('Could not publish actual server identity') from write_errors[0]


if __name__=='__main__':main()
