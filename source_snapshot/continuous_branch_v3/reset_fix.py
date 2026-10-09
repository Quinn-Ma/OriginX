"""Development-only stable counter geometry ordering; original source unchanged."""
import hashlib,inspect,textwrap

def install():
 from robocasa.models.fixtures.counter import Counter
 module=__import__(Counter.__module__,fromlist=['Counter'])
 source=inspect.getsource(Counter.get_reset_regions)
 old='valid_geoms = list(set(valid_geoms))'
 assert source.count(old)==1,'Counter implementation changed'
 whole=open(module.__file__,'rb').read()
 assert hashlib.sha256(whole).hexdigest()=='77f992d01aa1ae5f21ed7f170d0aab9c303c32148f4b427651c04912be8ce68a'
 changed=textwrap.dedent(source.replace(old,'valid_geoms = list(dict.fromkeys(valid_geoms))'))
 namespace={};exec(compile(changed,'<continuous-v3-stable-counter>','exec'),module.__dict__,namespace)
 Counter.get_reset_regions=namespace['get_reset_regions']
 return {'original_sha256':hashlib.sha256(whole).hexdigest(),'patched_method_sha256':hashlib.sha256(changed.encode()).hexdigest(),'on_disk_sources_changed':False,'development_only':True}
