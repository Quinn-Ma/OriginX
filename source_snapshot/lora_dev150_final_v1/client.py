"""Preserve the verified processor/RPC client and additionally pin adapter identity."""
from local_eval.remote_client import RemoteEvalClient, validate_endpoint
from .loader import read_binding, sha, require
from .server import check_lora_hello


class AdapterEvalClient(RemoteEvalClient):
    def __init__(self,repo,processor_path,local_port,expected_hello,**kwargs):
        self.expected_lora_hello=expected_hello
        super().__init__(repo,processor_path,local_port,expected_hello,**kwargs)
        try:check_lora_hello(self.wire.hello,expected_hello)
        except BaseException:
            self.close();raise
    def reset(self,seed):
        result=super().reset(seed)
        check_lora_hello(result,self.expected_lora_hello)
        return result


def validate_adapter_endpoint(manifest_path,parity_path,processor_path,*,binding_path,binding_sha256):
    b=read_binding(binding_path,binding_sha256)
    manifest=validate_endpoint(manifest_path,parity_path,processor_path)
    hello=manifest['hello'];specific=hello.get('lora_serving_identity',{})
    require(specific.get('binding_sha256')==binding_sha256
            and specific.get('arm')==b['arm']
            and specific.get('adapter_file_sha256')==b['adapter']['sha256']
            and specific.get('inference_arithmetic')==b['inference_arithmetic'],
            'Base-only identity cannot certify this adapter endpoint')
    import json
    from pathlib import Path
    parity=json.loads(Path(parity_path).read_text())
    check_lora_hello(hello,parity['model'])
    return manifest
