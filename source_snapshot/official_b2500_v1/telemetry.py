"""Observe the existing serialized forward and keep the original response bytes."""
import json
import os
from pathlib import Path
from multiplex_inference.core import SerialExecutor


class StageLoggingExecutor(SerialExecutor):
    def __init__(self,model,torch,identity,*,path,capacity=8):
        self.trace_path=Path(path)
        self.trace_path.parent.mkdir(parents=True,exist_ok=True)
        self.trace_stream=self.trace_path.open('x',buffering=1)
        try:super().__init__(model,torch,identity,capacity=capacity)
        except BaseException:
            self.trace_stream.close();raise

    def _handle(self,connection,payload):
        before=self.model.forward_calls
        result=super()._handle(connection,payload)
        after=self.model.forward_calls
        if after!=before:
            try:
                if after!=before+1:raise RuntimeError('Telemetry observed multiple model forwards')
                session=self._sessions[connection]
                row=dict(schema='stage_query_v2',server_instance=self.identity['server_instance'],
                    arm=self.identity['stage_serving_identity']['arm'],connection_id=connection,
                    seed=session['seed'],requests_since_reset=session['requests'],forward_index=after,
                    **self.model.last_stage_trace)
                self.trace_stream.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n')
            except BaseException:
                self._sessions[connection]['poisoned']=True
                raise
        return result

    def close(self):
        try:super().close()
        finally:
            if not self.trace_stream.closed:
                self.trace_stream.flush();os.fsync(self.trace_stream.fileno());self.trace_stream.close()
