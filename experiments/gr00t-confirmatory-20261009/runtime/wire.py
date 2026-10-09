"""Bounded JSON/typed-array RPC; never deserialize pickle or executable objects."""
import base64
import json
import math
import struct
import numpy as np
from .adapter import PROTOCOL

MAX_PAYLOAD = 4*1024*1024

def _pack(value):
    if isinstance(value,np.ndarray):
        if value.dtype not in (np.dtype('uint8'),np.dtype('float32')):
            raise ValueError('Wire arrays must be uint8 or float32')
        a=np.ascontiguousarray(value)
        return {'__array__':True,'dtype':a.dtype.str,'shape':list(a.shape),
                'data':base64.b64encode(a.tobytes()).decode('ascii')}
    if isinstance(value,dict):return {k:_pack(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_pack(v) for v in value]
    if value is None or type(value) in (str,int,float,bool):return value
    raise ValueError('Unsupported wire value')

def _unpack(value):
    if isinstance(value,dict):
        if '__array__' in value:
            if set(value)!={'__array__','dtype','shape','data'} or value['__array__'] is not True:
                raise ValueError('Malformed array envelope')
            if value['dtype'] not in ('|u1','<f4'):
                raise ValueError('Unsupported array dtype')
            shape=value['shape']
            if not isinstance(shape,list) or not 1<=len(shape)<=5 or any(type(n)is not int or not 0<n<=4096 for n in shape):
                raise ValueError('Invalid bounded array shape')
            dtype=np.dtype(value['dtype']);expected=math.prod(shape)*dtype.itemsize
            if expected>MAX_PAYLOAD:raise ValueError('Array exceeds payload bound')
            raw=base64.b64decode(value['data'],validate=True)
            if len(raw)!=expected:raise ValueError('Array byte count differs')
            return np.frombuffer(raw,dtype=dtype).reshape(shape).copy()
        return {k:_unpack(v) for k,v in value.items()}
    if isinstance(value,list):return [_unpack(v) for v in value]
    return value

def encode(value):
    payload=json.dumps(_pack(value),allow_nan=False,separators=(',',':')).encode()
    if not 0<len(payload)<=MAX_PAYLOAD:raise ValueError('Payload exceeds bound')
    return payload

def decode(payload):
    if not 0<len(payload)<=MAX_PAYLOAD:raise ValueError('Payload exceeds bound')
    def reject(value):raise ValueError('Nonfinite JSON number: '+value)
    return _unpack(json.loads(payload,parse_constant=reject))

def recv_exact(sock,length):
    pieces=bytearray()
    while len(pieces)<length:
        chunk=sock.recv(length-len(pieces))
        if not chunk:raise EOFError('Socket closed')
        pieces.extend(chunk)
    return bytes(pieces)

def receive(sock):
    length=struct.unpack('>I',recv_exact(sock,4))[0]
    if not 0<length<=MAX_PAYLOAD:raise ValueError('Invalid frame size')
    return recv_exact(sock,length)

def send(sock,payload):
    if not 0<len(payload)<=MAX_PAYLOAD:raise ValueError('Invalid frame size')
    sock.sendall(struct.pack('>I',len(payload))+payload)
