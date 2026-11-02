# BranchBox: MemBodied port for X-VLA

This directory releases the model mechanism and its unit tests. It adapts the
MemBodied/OpenPI architecture to X-VLA; it is not an independently invented
memory architecture or the official pi0 implementation. See NOTICE.txt,
FILE_ORIGINS.json, and code/PORT_MAPPING.md for attribution and adaptations.

The caller supplies an already-loaded X-VLA policy and its native preprocessing.
Construct `FullMemBodiedPolicy(loaded_policy.model)` from `code/full_policy.py`.
The wrapper widens that model in place; use a separate loaded object for the
original baseline. `normalization.py` supplies exact executed-action
normalization from the checkpoint processor.

The paired benchmark compares the original policy with the fully fine-tuned
memory policy and the same trained policy with its added memory history reset.
The external prompt controller remains active in that reset condition.
Checkpoint weights, dataset videos, private server paths, transfer incident
records, and machine-specific orchestration are not included in this core
source release. No success-rate claim is implied by this code package.

Dependencies used for the experiment include Python3.12, PyTorch2.8.0,
LeRobot0.6.1, and Transformers5.5.4. Native pretrained assets and their licenses
remain separate. Run the mechanism tests from `code/` with
`python -m unittest test_full_memory -v` in the matching environment.
