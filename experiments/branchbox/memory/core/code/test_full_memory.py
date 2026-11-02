"""CPU mechanism tests using the installed, unmodified X-VLA transformer."""
import copy
import unittest
import torch
from torch import nn
from lerobot.policies.xvla.soft_transformer import SoftPromptedTransformer
from full_memory import FullAssociativeMemory, FullMemorySpec, MemoryCarry
from full_policy import FullMemBodiedPolicy, widen_action_encoder


def small_spec(layers=3):
    return FullMemorySpec(width=32, vision_width=16, proprio_dim=20, layers=layers,
                          rank=8, memory_alpha=16, anchor_rank=4, vision_grid=16)


class DummyBase(nn.Module):
    def __init__(self, layers=3):
        super().__init__()
        self.transformer = SoftPromptedTransformer(hidden_size=32, multi_modal_input_size=16,
            depth=layers, num_heads=4, num_domains=2, dim_action=20, dim_propio=20,
            dim_time=8, len_soft_prompts=2, max_len_seq=64)
        self.vlm = nn.Module()
        self.vlm.language_model = nn.Module()


class FullMemoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        torch.set_num_threads(2)

    def test_official_gated_delta_formula(self):
        m = FullAssociativeMemory(small_spec())
        carry = MemoryCarry(torch.randn(2,3,8,8), torch.randn(2,32,16))
        h, a, v = torch.randn(2,3,32), torch.randn(2,5,7), torch.randn(2,16)
        result = m.write(carry,h,a,v)
        value = torch.cat([v,a.sum(1)],-1)
        for i,p in enumerate(m.layers):
            k=torch.tanh(p.key(h[:,i])); k=k/k.norm(dim=-1,keepdim=True).clamp_min(1e-12)
            alpha=torch.sigmoid(p.alpha(torch.cat([h[:,i],value],-1))+2)
            beta=torch.sigmoid(p.beta(torch.cat([h[:,i],value],-1))-1.5)
            old=carry.S[:,i]
            expected=alpha[...,None]*old+beta[...,None]*(p.value(value)-alpha*(old@k[...,None]).squeeze(-1))[...,None]*k[:,None]
            torch.testing.assert_close(result.S[:,i],expected)
        self.assertIs(result.anchor,carry.anchor)

    def test_spatial_anchor_exact_blocks_and_detached(self):
        m=FullAssociativeMemory(small_spec())
        patches=torch.arange(2*256*16,dtype=torch.float32).reshape(1,2,256,16).requires_grad_()
        out=m.extract_anchor(patches)
        expected=[]
        grids=patches.reshape(1,2,16,16,16)
        for camera in range(2):
            for row in range(4):
                for col in range(4):
                    expected.append(grids[:,camera,row*4:(row+1)*4,col*4:(col+1)*4].mean((1,2)))
        torch.testing.assert_close(out,torch.stack(expected,1))
        self.assertFalse(out.requires_grad)
        with self.assertRaises(ValueError): m.extract_anchor(torch.zeros(1,2,49,16))

    def test_state_query_pool_not_plain_patch_mean(self):
        m=FullAssociativeMemory(small_spec())
        patches=torch.randn(2,2,256,16,requires_grad=True)
        state=torch.randn(2,32,requires_grad=True)
        v=m.vision_encoding(patches,state)
        changed=m.vision_encoding(patches,state+torch.randn_like(state)*2)
        self.assertGreater((changed-v).abs().max().item(),1e-5)
        v.square().mean().backward()
        self.assertIsNone(patches.grad)
        self.assertGreater(state.grad.abs().sum().item(),0)
        self.assertGreater(m.pool_q.weight.grad.abs().sum().item(),0)

    def test_init_and_read_match_official_initialization(self):
        m=FullAssociativeMemory(small_spec())
        carry=m.initial_carry(torch.randn(2,2,256,16))
        self.assertGreater(m.S_init.std().item(),0)
        self.assertGreater(m.memory_token.std().item(),0)
        self.assertTrue(torch.equal(m.read(0,torch.randn(2,32),carry.S),torch.zeros(2,32)))
        torch.testing.assert_close(carry.S[0],carry.S[1],rtol=0,atol=0)

    def test_sum_actions_mask_and_inactive_write(self):
        m=FullAssociativeMemory(small_spec())
        carry=m.initial_carry(torch.randn(2,2,256,16))
        h,a,v=torch.randn(2,3,32),torch.randn(2,4,7),torch.randn(2,16)
        mask=torch.tensor([[True,True,False,False],[False,False,False,False]])
        padded=a.clone(); padded[~mask]=999
        result=m.write(carry,h,padded,v,mask,torch.tensor([True,False]))
        reference=m.write(MemoryCarry(carry.S[:1],carry.anchor[:1]),h[:1],a[:1,:2],v[:1])
        torch.testing.assert_close(result.S[:1],reference.S)
        torch.testing.assert_close(result.S[1],carry.S[1],rtol=0,atol=0)
        with self.assertRaises(ValueError): m.write(carry,h,a,v,mask)

    def test_literal_concat_widening_preserves_original_affine(self):
        encoder=DummyBase().transformer.action_encoder
        wide=widen_action_encoder(encoder,32)
        x=torch.randn(2,3,encoder.input_size); domain=torch.tensor([0,1])
        torch.testing.assert_close(encoder(x,domain),wide(torch.cat([x,torch.randn(2,3,32)],-1),domain),rtol=0,atol=0)
        self.assertTrue(torch.equal(wide.fc.weight.reshape(2,-1,32)[:,encoder.input_size:],torch.zeros(2,32,32)))

    def make_context(self,model):
        return model.context_from_features(torch.randn(1,4,16,requires_grad=True),
            torch.randn(1,4,16,requires_grad=True),torch.randn(1,20),torch.zeros(1,dtype=torch.long),
            torch.randn(1,2,256,16,requires_grad=True))

    def test_all24_layers_dedicated_tokens_native_decode_and_gradients(self):
        model=FullMemBodiedPolicy(DummyBase(24),small_spec(24)).eval()
        context=self.make_context(model); carry=model.init_carry(context)
        observed=[]
        handles=[block.attn.register_forward_pre_hook(lambda module,args: observed.append(args[0].shape[1]))
                 for block in model.base.transformer.blocks]
        output,h=model.predict(context,torch.randn(1,3,20),torch.tensor([.4]),carry)
        for hook in handles:hook.remove()
        self.assertEqual(output.shape,(1,3,20));self.assertEqual(h.shape,(1,24,32))
        self.assertEqual(observed,[3+4+4+2+2]*24)
        output.square().mean().backward()
        self.assertFalse(model.parameter_report()['frozen_parameter_names'])
        self.assertGreater(context.vlm_features.grad.abs().sum().item(),0)
        self.assertIsNone(context.patches.grad)
        for block in model.base.transformer.blocks:
            self.assertGreater(block.attn.qkv.weight.grad.abs().sum().item(),0)
        self.assertGreater(model.memory.state_proj.weight.grad.abs().sum().item(),0)

    def test_delayed_write_bptt_and_checkpoint_equivalence(self):
        model=FullMemBodiedPolicy(DummyBase(),small_spec()).eval()
        for p in model.memory.layers: nn.init.normal_(p.read_out.weight,std=.03)
        contexts=[self.make_context(model) for _ in range(3)]
        carry=model.init_carry(contexts[0]); previous=None
        for t,ctx in enumerate(contexts):
            if previous is not None:
                carry=model.advance(carry,previous,torch.randn(1,3,7),ctx)
            prediction,previous=model.predict(ctx,torch.randn(1,3,20),torch.tensor([.4]),carry)
        prediction.square().mean().backward()
        for layer in model.memory.layers:
            self.assertIsNotNone(layer.value.weight.grad)
            self.assertGreater(layer.value.weight.grad.abs().sum().item(),0)
        self.assertGreater(model.memory.S_init.grad.abs().sum().item(),0)
        self.assertGreater(contexts[0].vlm_features.grad.abs().sum().item(),0)
        # Same checkpointed/noncheckpointed model, dropout RNG restored.
        for p in model.parameters():p.grad=None
        model.train(); context=self.make_context(model); carry=model.init_carry(context)
        noise=torch.randn(1,3,20)
        torch.manual_seed(11); a,_=model.predict(context,noise,torch.tensor([.5]),carry)
        model.gradient_checkpointing=True
        torch.manual_seed(11); b,_=model.predict(context,noise,torch.tensor([.5]),carry)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        b.square().mean().backward()
        self.assertGreater(model.base.transformer.blocks[0].attn.qkv.weight.grad.abs().sum().item(),0)


if __name__=='__main__': unittest.main(verbosity=2)
