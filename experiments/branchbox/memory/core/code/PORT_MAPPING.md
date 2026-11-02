# Full MemBodied main-architecture port to trained X-VLA-P

Reference: `declare-lab/MemBodied` commit `e268134a9e5c4b99a4607b92f466f25d784b3f26`, `RMBench/policy/pi0/src/openpi/models/pi0_mem_vector.py` and `_vector_model()` in `training/config.py`. This is a PyTorch/X-VLA **port of the main architecture**, not a claim to reproduce the official pi0 LIBERO configuration, numerical results, or mask geometry.

## Preserved mechanisms

- All 24 action-expert layers contain one rank-128 associative matrix, with read scale 256/128. Learned matrix initialization is Normal(0,.02), not zero. A separate trainable memory token is Normal(0,.02); read projection and scalar read gate affine parameters are zero initialized (sigmoid gate starts .5).
- Each layer reads using its pre-attention-normalized dedicated **state** token. The memory vector is added to the pre-attention-normalized dedicated **memory** token before attention. Its post-final-FFN state token supplies the later write key. Neither uses an action-token mean.
- The current proprioceptive state projects to a state token. That token queries all spatial tokens in each camera using learned scaled dot-product attention; the resulting camera summaries are averaged. This is not uniform spatial averaging. Raw visual patches are detached only on the memory branches.
- The true episode-initial raw images are retained and encoded with native Florence geometry. Each7×7 clean-camera patch grid is adaptively pooled into16 spatial cells on a4×4 grid. Two clean physical cameras yield32 fixed anchor tokens. A rank64 cross-attention read queries this anchor from the current state-conditioned visual summary; its result plus current summary conditions action tokens through a literal widened affine concat input. The7→4 pooling is an explicit user-directed backbone adaptation, not official16→4 geometry.
- The stored value concatenates the **next observed** visual summary with the **sum** of the actually executed normalized controls. Vector alpha/beta gates use offsets +2/−1.5 and the official row-wise gated delta formula. No predicted but unexecuted chunk tail, success label, reward, or future observation is passed into the current action prediction.
- Ordinary VLA visual, language, and action weights are trainable. The caller also optimizes the existing BB4 stage/target heads with the original auxiliary objective. Original Florence text decoder is absent by native X-VLA design; no nonexistent decoder is claimed to be trained.

## Necessary backbone and cadence mappings

1. Native X-VLA uses 30 action tokens, encoded main-image/text, prompt/wrist tokens, and32 soft prompts with fully bidirectional expert attention. Two genuine tokens are appended after existing prompts; no prompt slot is repurposed. Original token positions/action slicing remain unchanged. Unlike pi0's suffix-causal attention, native X-VLA state tokens may attend noisy action tokens. Adding tokens changes attention even at zero read projection; baseline identity is **not** claimed.
2. Native224px Florence yields7×7 patches rather than SigLIP16×16. The user explicitly chose retaining original first-frame images instead of adding a512px branch. Current memory features reuse the same native encoding as the ordinary VLA path, with a detach only at the memory branch. Native learned2D/temporal positional processing is retained; only the leading global-summary token is discarded from50 projected tokens. Adaptive7→4 spatial pooling preserves scene-anchor functionality but differs from the official nonoverlapping16→4 blocks. The first-frame anchor is encoded once at native224 resolution from original physical-camera images, no synthetic prompt camera. There is no feature upsampling or second current-image vision pass.
3. Native action/proprio/time concatenation feeds a domain-specific affine map, unlike pi0's action/time MLP. That same native map is widened by1024 anchor channels; all original coefficients/biases are copied exactly and only new anchor coefficients start zero. All coefficients train. Domain rows absent from the dataset naturally receive no update.
4. Native clean-action prediction is preserved:7 real action dimensions padded to20 **before** adding20-dimensional noise; masked clean-action MSE on the first7 output dimensions. This is not pi0's velocity objective. Native inference remains10 fixed-point clean-action updates.
5. There is one write slot per actually executed30-control query. Each query uses the same S through all10 denoising iterations, saves final per-layer state tokens, executes the queued controls, receives the next image, and then writes once. This differs from official rollout configurations that maintain multiple action slots or store predicted rather than observed executed chunks.
6. Python state is explicit: `MemoryCarry.S[B,24,128,128]`, `anchor[B,32,1024]`, previous h `[B,24,1024]`. No internal training detach. Caller performs full BPTT inside each prescribed6-query window, initializes S from learned S0 at window boundaries, and supplies the true episode-first-frame anchor even for windows beginning mid-episode. Inference carries state throughout an episode; all-history reset clears S and pending actions/h and resets the anchor from the current observation when that control is requested.
7. Full fine-tuning rather than the main reference's LoRA is user requested. New affine initializations match LeCun truncated-normal variance; different PyTorch/JAX RNG implementations do not imply bit-identical initial tensors. Memory math is FP32 under BF16 expert autocast for numerical stability.

## API

```python
port = FullMemBodiedPolicy(loaded_policy.model, gradient_checkpointing=True)
# Each argument below is already processed by the unchanged checkpoint processor.
context = port.encode_observation(input_ids, pixel_values224, image_mask,
                                  proprio20, domain_id)
anchor_patches = port.encode_memory_patches(true_episode_initial_clean_pixels224)
carry = port.init_carry(context, episode_anchor_patches=anchor_patches)
clean20, h = port(context, noisy20, time, carry)
# Only after executing controls and observing the next frame:
next_carry = port.advance(carry, h, executed_controls7, next_context,
                          action_mask=executed_mask, batch_mask=valid_sequence_row)
```

`predict` directly calls the native action decoder, preserving the existing BB4 decoder-input hook. `sample` is inference-only. Nonreentrant activation checkpointing covers the normal VLM and every expert layer, preserving RNG and BPTT; it does not truncate recurrence. Constructing the port widens its supplied base model in place, so the matched baseline must load its own fresh checkpoint object.
