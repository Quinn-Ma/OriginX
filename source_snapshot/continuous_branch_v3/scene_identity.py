import hashlib,json
import numpy as np

def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,default=lambda v:v.tolist() if hasattr(v,'tolist') else str(v)).encode()).hexdigest()
def array(x):
 x=np.asarray(x);return hashlib.sha256(str((x.shape,x.dtype.str)).encode()+x.tobytes()).hexdigest()
def stable(v,depth=0):
 if depth>8:return type(v).__name__
 if isinstance(v,(str,int,float,bool,type(None))):return v
 if isinstance(v,np.ndarray):return {'array':array(v)}
 if isinstance(v,dict):return {str(k):stable(z,depth+1) for k,z in sorted(v.items(),key=lambda q:str(q[0])) if k not in ('sim','robots','_model')}
 if isinstance(v,(list,tuple)):return [stable(z,depth+1) for z in v]
 if hasattr(v,'__dict__') and ('controller' in type(v).__module__):return stable(vars(v),depth+1)
 return type(v).__module__+'.'+type(v).__name__
def extra_identity(core):
 return dict(qpos=array(core.sim.data.qpos),qvel=array(core.sim.data.qvel),ctrl=array(core.sim.data.ctrl),rng=canonical(core.rng.bit_generator.state),ep_meta=canonical(core.get_ep_meta()),controllers=canonical([stable(robot.composite_controller) for robot in core.robots]))
