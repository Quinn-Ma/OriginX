"""Pure safe admission: eight inventory entries, ECC0,12GiB/model+4GiB/card."""
def choose_layout(health,max_models=21):
 assert set(health)==set(range(8)),'Eight physical GPU identities required'
 assert type(max_models) is int and 1<=max_models<=48,'Capacity cap1..48'
 counts={g:min(6,max(0,(v['free_mib']-4096)//12288)) if v['ecc']=='0' else 0 for g,v in health.items()}
 layout=[dict(arm='B',gpu_index=g,gpu_uuid=health[g]['uuid'],replica=i,key=f'B-gpu{g}-r{i}',port=16600+g*6+i) for i in range(6) for g in range(8) if i<counts[g]][:max_models]
 assert layout,'No available healthy capacity; foreign jobs preserved'
 return layout,{g:sum(x['gpu_index']==g for x in layout) for g in range(8)}
