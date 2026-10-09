# External startup and finalization corrections

Before any formal rollout, read-only review identified a process-start race in immediate `/proc` argv inspection. The unexecuted first full launcher is preserved. `start_full_admitted_v2.py` replaces only the external launch boundary; the frozen model service and campaign runtime, all scientific parameters and source pins remain unchanged.

Each newly spawned service is held through a parent-owned Popen object and Linux pidfd before identity binding. Only the four expected expansion commands are intercepted by a module-local proxy. Stable full argv/cwd and immutable start identity must match before handing ownership to the original service code. Campaign handoff additionally waits for its own complete owner receipt. Errors preserve partial directories and drain only precisely owned pending children and known profiles. Shared Unix identity is never sufficient authority to terminate a process.

Validation: 21 tests passed locally with one Linux-only test skipped; on the actual Linux host, 21 tests passed including the real child/pidfd test, with the Windows driver test skipped. The read-only actual admission precheck passed at 2026-10-09 20:02:46 UTC. This precheck does not mean a formal rollout occurred.

The finalizer was replaced only while waiting for a campaign, using a held Windows process handle and exact creation identity. Earlier sources and replacement receipts remain in this directory. `finalize_full_v3.py` requires explicit integer zero CLI calls before treating an empty invocation ledger as empty; it never scans unrelated Codex processes with an empty path. Seven zero-ledger tests passed. It starts no inference, training or public publication and only consolidates terminal evidence.
