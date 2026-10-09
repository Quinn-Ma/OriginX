"""Preserve the verified processor/RPC client and additionally pin adapter identity."""
from local_eval.remote_client import RemoteEvalClient, validate_endpoint
from .loader import read_binding, sha, require
from .loader import check_stage_hello


class StageEvalClient(RemoteEvalClient):
    def __init__(self,repo,processor_path,local_port,expected_hello,**kwargs):
        self.expected_stage_hello=expected_hello
        super().__init__(repo,processor_path,local_port,expected_hello,**kwargs)
        try:check_stage_hello(self.wire.hello,expected_hello)
        except BaseException:
            self.close();raise
    def reset(self,seed):
        result=super().reset(seed)
        check_stage_hello(result,self.expected_stage_hello)
        return result


def validate_stage_endpoint(manifest_path,parity_path,processor_path,*,binding_path,binding_sha256):
    b=read_binding(binding_path,binding_sha256)
    manifest=validate_endpoint(manifest_path,parity_path,processor_path)
    hello=manifest['hello'];specific=hello.get('stage_serving_identity',{})
    require(specific.get('binding_sha256')==binding_sha256
            and specific.get('arm')==b['arm']
            and specific.get('branch_file_sha256')==b['branch']['sha256']
            and specific.get('branch_identity')==b['branch_identity']
            and specific.get('online_stage_conditioning') is True
            and specific.get('no_label_actor_inputs') is True
            and specific.get('frozen_A_all_parameters') is True
            and specific.get('inference_arithmetic')==b['inference_arithmetic'],
            'A-only identity cannot certify an online stage endpoint')
    import json
    from pathlib import Path
    parity=json.loads(Path(parity_path).read_text())
    require(parity.get('stage_path_checked_on_every_forward') is True and parity.get('stage_full_forward_calls',0)>=12,'Missing actual stage direct/socket forwards')
    require(manifest.get('stage_binding_sha256')==binding_sha256,'Server was loaded from a different branch binding')
    check_stage_hello(hello,parity['model'])
    return manifest
