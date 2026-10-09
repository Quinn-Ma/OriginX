# Astra rescue evidence

The fixed cohort consists of 1,004 unique historical B2000 failures. Both arm attempts were scheduled for each case; all 2,008 are terminal. The final classification is 56 rescued / 813 not rescued / 28 unknown / 107 initial-state deviation, with M=874 matched completed pairs and E=869 verified applied-Astra pairs. The original 1,496/2,500 benchmark stays unchanged.

`case-classification.json` and `cli-usage.json` are byte-identical snapshots produced by the local finalizer before publication. Their `publicly_uploaded: false` fields describe that creation-time state; this release publishes these exact snapshots without rewriting their provenance. `experiment_complete: false` preserves the unresolved outcomes even though all attempts have ended. A final independent audit and publication receipt accompany them.

The archived campaign bundles contain per-arm results and native initial-state checks, prefix parity, RGB request images, Astra request/response and actual application records, runtime configuration/source hashes, logs, and process ownership evidence. Internal experiment paths are retained for provenance. Request tokens are per-case binding nonces, not account credentials. Model weights remain on Hugging Face.

Download the three campaign bundles from [the rescue evidence release](https://github.com/Quinn-Ma/OriginX/releases/tag/v1.1.0-astra-rescue):

| Bundle | SHA-256 |
|---|---|
| `astra-rescue-v1-evidence.tar.gz` | `39d033c56789c76ae511c8e197370b82767a192bdc87ef9f70cd3638c848ddbb` |
| `astra-rescue-v2-evidence.tar.gz` | `a293bad10e191f8126068c437904c27b320094fe8b52ad6784eea3cfb72f61aa` |
| `astra-rescue-pilot-evidence.tar.gz` | `ed743e883ae06dc734769fa5cf6c86a946e3f328645a57889ef49719683360a5` |

The pilot indices 48/97/249/345 are development cases, separately 0/4 rescued; they were not removed from the main study or counted twice. v1 contributes its fixed first 36 cases and v2 the remaining 968. Neither deviations nor technical failures were selectively retried.
