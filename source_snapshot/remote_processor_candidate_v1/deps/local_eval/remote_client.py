"""Official processor/action path over an owned SSH localhost tunnel.

Remote PID evidence is provenance only, never looked up in local /proc.
The frozen server instance and model bytes are bound by the live hello.
"""
import importlib.util
import inspect
import json
from pathlib import Path
import socket
import pickle
import struct
import time

from multiplex_inference.client import WireClient
from multiplex_inference.server import sha256, MAX_PAYLOAD, recv_exact
from multiplex_inference.core import PROTOCOL

ENTRY_SHA='18d70fade4c990210dc5b9c34285c8030eac142d9c332066c73335e1c0aaa28e'
HELLO_FIELDS=('server_instance','protocol','exclusive_connection','shared_model','isolated_rng_stream',
 'serial_forward','rng_isolation','model_path','model_config_sha256','model_assets_sha256',
 'official_server_sha256','multiplex_sources_sha256','model_config_runtime_sha256',
 'model_execution_thread_name','model_execution_thread_ident','torch_version',
 'cuda_visible_device_count','full_request_only','inference_optimization')


def check_hello(actual, expected):
    if any(actual.get(k)!=expected.get(k) for k in HELLO_FIELDS):
        raise ValueError('SSH endpoint hello differs from pinned remote server identity')


def validate_endpoint(manifest_path, parity_path, processor_path, *, source_package=None):
    manifest=json.loads(Path(manifest_path).read_text());hello=manifest['hello']
    if manifest.get('ready') is not True or len(manifest.get('servers',[]))!=1:
        raise ValueError('One ready remote multiplex server is required')
    if (hello.get('protocol')!=PROTOCOL or hello.get('exclusive_connection') is not False
            or hello.get('isolated_rng_stream') is not True or hello.get('serial_forward') is not True
            or hello.get('inference_optimization')!='none' or hello.get('cuda_visible_device_count')!=1
            or hello.get('model_execution_thread_name')!='xr1-serial-forward'
            or not hello.get('model_config_runtime_sha256') or not hello.get('server_instance')):
        raise ValueError('Remote protocol/model execution identity differs')
    if hello['multiplex_sources_sha256']!=manifest['source_sha256']:
        raise ValueError('Remote manifest source identity differs')
    if source_package is None:
        import multiplex_inference.core as core
        source_package=Path(core.__file__).parent
    for name,digest in manifest['source_sha256'].items():
        if Path(name).name!=name or not name.endswith('.py') or sha256(Path(source_package)/name)!=digest:
            raise ValueError('Frozen multiplex transport differs from remote parity source')
    if sha256(parity_path)!=manifest['parity_sha256']:
        raise ValueError('Copied remote parity report differs')
    parity=json.loads(Path(parity_path).read_text())
    if (parity.get('passed') is not True or len(parity.get('trace',[]))!=6
            or not all(r['full_action_exact'] and r['all_rng_exact'] for r in parity['trace'])
            or parity.get('ambient_rng_unchanged') is not True
            or parity['source_sha256']!=manifest['source_sha256']
            or parity['model']['model_assets_sha256']!=hello['model_assets_sha256']
            or parity['trace'][0]['actual_rng']['model_config_runtime_sha256']!=hello['model_config_runtime_sha256']
            or parity['model']['official_server_sha256']!=hello['official_server_sha256']):
        raise ValueError('Remote parity does not cover admitted model/source/runtime')
    processor=Path(processor_path).resolve(strict=True)
    for name,digest in hello['model_assets_sha256'].items():
        if Path(name).name!=name:raise ValueError('Unsafe model asset name')
        if not name.endswith('.safetensors') and sha256(processor/name)!=digest:
            raise ValueError('Local processor/code/config asset differs from remote weights release')
    if sha256(processor/'config.json')!=hello['model_config_sha256']:
        raise ValueError('Local processor/model configuration differs')
    return manifest


class MeasuredWireClient(WireClient):
    """Same pickle/framing bytes; records actual sizes and RPC wall time, no compression."""
    def __init__(self,*args,telemetry_path=None,**kwargs):
        self.telemetry=open(telemetry_path,'x') if telemetry_path is not None else None
        self.pending=None;self.inference_calls=0;self.request_bytes=0;self.response_bytes=0
        try:super().__init__(*args,**kwargs)
        except BaseException:
            if self.telemetry:self.telemetry.close()
            raise
    def send(self,data):
        start=time.perf_counter();payload=pickle.dumps(data,protocol=pickle.HIGHEST_PROTOCOL)
        if not 0<len(payload)<=MAX_PAYLOAD:raise ValueError('Request exceeds payload bound')
        encoded=time.perf_counter();fields={}
        control='_multiplex_control' in data
        if not control:
            for key,value in data.items():
                if hasattr(value,'shape'):
                    if hasattr(value,'device') and str(value.device)!='cpu':raise ValueError('Local processor request must remain CPU-only')
                    fields[key]=dict(shape=list(value.shape),dtype=str(value.dtype),
                        unencoded_tensor_bytes=int(value.numel()*value.element_size()) if hasattr(value,'numel') else int(value.nbytes))
        self.socket.sendall(struct.pack('>I',len(payload))+payload)
        self.pending=dict(start=start,serialized_seconds=encoded-start,sent_at=time.perf_counter(),
            request_payload_bytes=len(payload),request_framing_bytes=4,fields=fields,
            kind='control' if control else 'inference',control=data.get('_multiplex_control'),
            encoding=f'pickle_protocol_{pickle.HIGHEST_PROTOCOL}_official_CPU_processor_tensors',lossy_compression=False)
    def receive(self):
        length=struct.unpack('>I',recv_exact(self.socket,4))[0]
        if not 0<length<=MAX_PAYLOAD:raise ValueError('Response exceeds payload bound')
        payload=recv_exact(self.socket,length);received=time.perf_counter();result=pickle.loads(payload);finished=time.perf_counter()
        row=self.pending;self.pending=None
        if row is not None:
            start=row.pop('start');sent=row.pop('sent_at')
            row.update(response_payload_bytes=length,response_framing_bytes=4,
                send_seconds=sent-start-row['serialized_seconds'],response_wait_seconds=received-sent,
                deserialize_seconds=finished-received,rpc_wall_seconds=finished-start,
                latency_scope='SSH plus transport plus server queue; inference rows also include full model forward, not isolated network RTT')
            if row['kind']=='inference':
                self.inference_calls+=1;self.request_bytes+=row['request_payload_bytes']+4;self.response_bytes+=length+4
            if self.telemetry:self.telemetry.write(json.dumps(row,allow_nan=False)+'\n');self.telemetry.flush()
        return result
    def close(self):
        try:super().close()
        finally:
            if self.telemetry and not self.telemetry.closed:self.telemetry.close()


class RemoteEvalClient:
    """One persistent logical-worker RNG stream; no reconnect/reseed after failure."""
    def __init__(self, repo, processor_path, local_port, expected_hello, *, timeout=180, telemetry_path=None):
        root=Path(repo).resolve(strict=True);source=root/'eval_robocasa365/entry.py'
        if sha256(source)!=ENTRY_SHA:raise ValueError('Official evaluator source changed')
        spec=importlib.util.spec_from_file_location('_local_remote_official_entry',source)
        entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
        original=entry.Client
        if Path(inspect.getfile(original)).resolve()!=root/'deploy/client.py':
            raise ValueError('Official processor client imported from another repository')
        class BoundedClient(original):
            def _connect_with_retry(self, max_retries=None, retry_interval=1):
                self.client_socket=socket.create_connection((self.host,self.port),timeout=timeout)
                self.client_socket.settimeout(timeout)
        entry.Client=BoundedClient
        self.eval=entry.EvalClient(str(Path(processor_path).resolve()),'127.0.0.1',local_port,'robocasa365',.95)
        self._initialized=False
        try:
            self.wire=MeasuredWireClient(local_port,timeout=timeout,_socket=self.eval.client.client_socket,telemetry_path=telemetry_path)
            check_hello(self.wire.hello,expected_hello)
            # Official processor request and decode_action are unchanged.
            self.eval.client._send_with_length_prefix=self.wire.send
            self.eval.client._recv_with_length_prefix=self.wire.receive
        except BaseException:
            self.close();raise
    def reset(self,seed):
        if self._initialized:raise ValueError('Only one RNG reset per logical worker connection')
        result=self.wire.reset(seed);self._initialized=True;return result
    def infer(self,*args,**kwargs):
        if not self._initialized:raise RuntimeError('Initialize logical worker RNG first')
        return self.eval.infer(*args,**kwargs)
    def close(self):
        try:
            if hasattr(self,'wire'):self.wire.close()
        finally:self.eval.close()
