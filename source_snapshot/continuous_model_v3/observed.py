"""Unchanged observed-only pool contract copied from joint_visual_stage_v1.
No parameters, cache files, future labels, or oracle inputs.
"""
import torch

def require(ok,message):
    if not ok:raise ValueError(message)

class ObservedFeaturePool:
    def pool_features(self, hidden, input_ids, attention_mask, video_grid_thw,
                      state, video_token_id, spatial_merge_size):
        require(isinstance(hidden,torch.Tensor) and hidden.ndim==3 and
                hidden.shape[-1]==self.hidden_dim and hidden.is_floating_point(), 'Expected hidden[B,S,D]')
        require(hidden.device.type!='meta', 'Cannot run on unmaterialized meta tensors')
        B,S,D=hidden.shape
        require(B>0 and S>0, 'Empty hidden sequence')
        integers=(torch.int8,torch.int16,torch.int32,torch.int64,torch.uint8)
        require(isinstance(input_ids,torch.Tensor) and input_ids.shape==(B,S) and input_ids.dtype in integers, 'Expected integer input_ids[B,S]')
        require(isinstance(attention_mask,torch.Tensor) and attention_mask.shape==(B,S)
                and attention_mask.dtype in integers+(torch.bool,), 'Expected binary attention_mask[B,S]')
        require(input_ids.device==attention_mask.device==hidden.device, 'Hidden/token/mask device mismatch')
        require(type(video_token_id) is int and video_token_id>=0, 'Explicit video token ID required')
        require(type(spatial_merge_size) is int and spatial_merge_size>0, 'Positive spatial merge size required')
        require(isinstance(video_grid_thw,torch.Tensor) and video_grid_thw.dtype in integers and
                tuple(video_grid_thw.shape) in ((3*B,3),(B,3,3)), 'Expected exactly3 camera grids per batch row')
        require(isinstance(state,torch.Tensor) and state.shape==(B,4,60) and state.is_floating_point()
                and state.device==hidden.device, 'Expected on-device state[B,4,60]')
        require(torch.isfinite(state).all() and (state[:,:,14:]==0).all(), 'Finite state with original14:60 zero padding required')
        ids=input_ids.detach().cpu().tolist();masks=attention_mask.detach().cpu().tolist()
        grids=video_grid_thw.detach().cpu().reshape(B,3,3).tolist();pooled=[]
        for b in range(B):
            require(all(v in (0,1) for v in masks[b]), 'Mask must be binary')
            valid=[i for i,v in enumerate(masks[b]) if v]
            require(valid and valid==list(range(valid[0],valid[-1]+1)), 'Only contiguous real tokens with left/right padding are supported')
            require(not any(tok==video_token_id and not masks[b][i] for i,tok in enumerate(ids[b])), 'Masked/future video tokens are not admitted')
            require(ids[b][valid[-1]]!=video_token_id, 'Full task prompt context must follow video tokens')
            # The official processor separates each temporal patch group by
            # timestamp/vision delimiter tokens; contiguous runs must agree.
            runs=[]
            for i in valid:
                if ids[b][i]==video_token_id:
                    if runs and runs[-1][-1]==i-1:runs[-1].append(i)
                    else:runs.append([i])
            cursor=0;camera=[]
            for T,H,W in grids[b]:
                require(T==2, 'Exactly4 observed frames / temporal_patch_size2 =>2 temporal groups per camera')
                require(H>0 and W>0 and H%spatial_merge_size==0 and W%spatial_merge_size==0, 'Invalid spatial grid/merge divisibility')
                n=(H//spatial_merge_size)*(W//spatial_merge_size)
                group=runs[cursor:cursor+T]
                require(len(group)==T and all(len(run)==n for run in group), 'Video tokens disagree with camera/temporal grid boundaries')
                camera.append(hidden[b,group[-1],:].mean(dim=0));cursor+=T
            require(cursor==len(runs), 'Extra/future camera or temporal groups')
            features=torch.cat(camera+[hidden[b,valid[-1],:],state[b,:,:14].reshape(-1).to(hidden.dtype)],dim=0)
            require(torch.isfinite(features).all(), 'Nonfinite selected observed features')
            pooled.append(features)
        return torch.stack(pooled,dim=0)
