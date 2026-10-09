# Independent B/base service groups

The external helper `multigpu_services_v1.py` is infrastructure only. It starts no evaluation, broker, training, or Astra invocation. Copy its exact verified bytes to the remote project root before invoking it. Frozen runtime sources remain unchanged.

```
envs/training/bin/python -B multigpu_services_v1.py --check --gpu 3
envs/training/bin/python -B multigpu_services_v1.py --run --gpu 3 --lease-seconds 10800
envs/training/bin/python -B multigpu_services_v1.py --check --gpu 6
envs/training/bin/python -B multigpu_services_v1.py --run --gpu 6 --lease-seconds 10800
```

Each `--run` is a long-lived guardian. A root-level launcher must owner-pin that guardian and retain its launch logs. GPU6 must first become empty after exact old-worker/model cleanup. GPU3 and GPU6 each require the pinned UUID, ECC zero, no foreign occupants, and at least 76 GiB free initially. Every further model load rechecks ownership and the remaining memory allowance. Existing group output refuses restart or overwrite.

Each group has four B models and two base models, six clients per model. GPU3 uses ports 28000–28005; GPU6 uses ports 28010–28015. Outputs are:

- `results/originx-confirmatory-multigpu-services-20261009-v1/gpuN/owner.json`
- `.../services-ready.json`: model inventory; this alone is not completed admission.
- `.../probes/confirmatory-v1/result.json`: exact action/RNG replay, 36 streams, 108 forwards, at least 360 seconds on the same sockets.
- `.../ready.json`: successful group admission binding both inventory and probe hashes.
- `.../spawns/`: durable pending-owner, bounded two-sample identity handshake, stdout/stderr for every own child.
- `.../first-error.json`, `cleanup.json`, `finished.json`: preserved terminal evidence.

The controller may combine two admitted groups, then produce its separate combined service admission. The original runner constructs policy clients directly from each actual profile. The frozen `services.make_client` is GPU6-specific and is not an API for the new GPU3 profiles.

To release a group, first stop all assigned workers and wait for their sockets to close. Create its previously absent `release-request.json`:

```json
{
  "schema": "originx_multigpu_service_release_v1",
  "services_ready_sha256": "ACTUAL_GROUP_SERVICES_READY_SHA256",
  "all_assigned_workers_stopped": true
}
```

The guardian still independently checks owner identities and connections before signalling its own held child handles. A 10800-second lease stops admitting new work after 9000 seconds, allowing 1800 seconds for drain. A live socket or changed owner at cleanup causes explicit preservation/blockage, never an aborted rollout or a signal to another user's process. The controller must react to `draining.json`; a hard expiry with active clients does not magically terminate them. Shorter leases are accepted from 1200 seconds with at most half the lease reserved for drain.

Verified helper SHA256: `9504e17868783952a17181c95102ff224b7c92b1f558185ac8ce9714741864e5`. New offline helper tests: 14 on Windows and 14 on actual WSL Linux. Reused tracked-child tests: 21 on each platform plus one platform-specific skip; Linux exercised a real child, pidfd, transient empty argv and reap. No remote service or rollout was started during these tests.
