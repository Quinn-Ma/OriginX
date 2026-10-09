# Independent executor v2: process-start race correction

The development-v1 attempt is retained unchanged, including its original `all_children_drained=false` completion and later independent terminal audit. It is an infrastructure attempt, not a scored confirmation result. No formal seeds, model weights, intervention parameters or assistant prompt were selected from that attempt.

## Boundary

Study namespace: `originx_gr00t_confirmation_20261009_v2`. Actual admitted model-service namespace and asset directory: `originx_gr00t_confirmation_20261009` (unchanged). Development output: `results/originx-gr00t-confirmatory-development-20261009-v2`. Formal output remains `results/originx-gr00t-confirmatory-20261009-v1` and is still subject to independent development admission and shared quota debit.

The new package has **14 executable modules**: `__init__.py`, `adapter.py`, `wire.py`, `core.py`, `server.py`, `client.py`, `parity_probe.py`, `runner.py`, `assistance.py`, `campaign.py`, `aggregate.py`, `broker.py`, `capture_native_fixtures.py`, and new `processes.py`.

Seven service modules are exact bytes from the admitted remote `engineering/source-v2` archive, whose identity digest is `90811c0ed4668f634c2beb04ba13bc208ab120a5391aa972c30b67e8b07bee93`. Two local working copies had CRLF line endings while admitted remote files had LF; only the new v2 copies were replaced with the exact archived bytes. This preserves byte identity instead of assuming equivalent line endings. Original local/remote v1 files are not changed.

The source freeze remains schema `originx_gr00t_source_freeze_v1`, but `namespace` is the study v2 name, `source_sha256` covers all 14 package-root Python modules, and `broker_source_sha256` pins the independent v2 broker. Config explicitly contains `service_namespace` equal to the old service namespace and pins both its seven actually executed source files and the new study source files. Worker owners must be v2; model owners must be the old service namespace. Aggregate validates both separately.

## Correction

`TrackedChild` is registered immediately after `Popen` returns, before any `/proc` inspection. It acquires a Linux pidfd while the direct child is unreaped, binds parent PID and immutable process-start ticks, and waits at most five seconds for two consecutive matching complete argv/cwd observations. An empty or transient cmdline is recorded and retried; it is never accepted as an owner or used as kill authority.

Every exception or startup timeout retains this pending child handle. Termination uses the same pidfd and birth identity even when complete argv binding never succeeded; the parent waits/reaps every child and retains a `spawns/<episode_id>.json` handshake receipt. SIGTERM followed by bounded SIGKILL is restricted to the exact owned spawn. Short-lived children remain known startup failures and are reaped. PID/start/parent changes prevent signaling. No process-name or username cleanup occurs.

Runner and campaign child launches both use this mechanism. An externally signaled rollout supervisor brings its worker deadline forward so its parent can drain it without orphaning workers. Campaign service cleanup is blocked when any campaign child or worker cleanup remains unconfirmed. The original denominator and unknown records are retained.

Schemas for episode, assistance, raw GR00T transport and Astra receipts remain unchanged. The opaque request IDs are namespaced to v2; case IDs, two disclosed development seeds, all 2,500 formal seeds, arm allocation/order, model settings, trigger, action horizon, task horizon, exact assistant prompt and shared budget remain unchanged.

## Verification

Local suite: **88 passed, one Linux-only test skipped on Windows**. Coverage includes the original observed fourth-child empty-cmdline failure, exception before the first complete identity, permanently empty identity timeout, wrong birth identity, initial pidfd attachment failure, ignored SIGTERM, and actual short-lived Python processes.

The Linux-only behavior was additionally verified through six real, disposable CPU subprocess checks on the execution host: short exit; normal stable identity and pidfd drain; five injected empty cmdline reads on a live child; permanent empty cmdline timeout; pre-bind inspection exception; and ignored TERM followed by same-pidfd KILL. **All six passed, all owned children reaped**, with no GPU loading, rollout or remote project file write. The receipt is `work/confirmatory_20261009/gr00t_v2_spawn_linux_regression.json`, bound to the exact `processes.py` SHA.

No development or formal rollout was launched by this repair task. Root must complete the new six-client 360-second parity for the newly admitted old-namespace service (port 27902), freeze/stage this v2 source, and launch the four development outcomes. Successful development releases that exact service; later formal admission requires its own live service/parity and the complete shared resource debit.
