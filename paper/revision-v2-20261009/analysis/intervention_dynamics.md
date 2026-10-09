# Two additional retrospective analyses

No new experiments, training, remote jobs, or assistant calls. All sources are immutable existing evidence.

## 1. Immediate action divergence

Across all869 verified eligible pairs, the control and assisted policy-query records have identical state and RGB fingerprints at the trigger, different instruction hashes, and different returned action hashes immediately at that query. The action-hash onset lag is0 steps in869/869; missing0. Of these,56 are strict rescues and813 remain failures.
This rules out an unchanged returned action tensor at intervention as the explanation for the813 non-rescues. It does not measure action magnitude, show that the physical response was useful, prove semantic grounding, or isolate Astra-specific reasoning. Hashes cannot reconstruct floating-point actions. Retrospective matching does not randomize latent service/arithmetic effects.

Producer verified against immutable source pins and AST: AssistanceClient.infer obtains action=client.infer(...), then hashes only array shape, dtype string and raw bytes. Instruction is a separately hashed field. The continuation inherits infer without override. Source line numbers and cryptographic pins are in JSON; action_hash is not a combined instruction/request fingerprint.

## 2. Completion timing within the original budget

For each rescued case, define u=(success_step-trigger)/(horizon-trigger). Median u=0.5167 (51.67% of remaining steps), IQR0.3045–0.7741, range0.1002–0.9733.
Median steps from intervention to success=361.0; mean=466.21. These summaries condition on56 observed rescues.

| Remaining action budget consumed | Confirmed rescues | Fraction of E869 | Fraction of R56 | Fraction of N1004 |
|---:|---:|---:|---:|---:|
| 25% | 11 | 1.27% | 19.64% | 1.10% |
| 50% | 27 | 3.11% | 48.21% | 2.69% |
| 75% | 41 | 4.72% | 73.21% | 4.08% |
| 90% | 49 | 5.64% | 87.50% | 4.88% |
| 100% | 56 | 6.44% | 100.00% | 5.58% |

15/56 rescues finish in the last quarter of their remaining action budget, and7/56 require more than90%. Thus a short continuation window would miss some successful recorded recoveries; these observations do not establish that an earlier assistance trigger would improve outcomes.
There are no missing source records among E869. All813 eligible non-rescues reach the original horizon. The28 unknown and107 reset-deviation cases remain outside eligible timing analysis and inside the N1004 accounting. Original1496 successes are not reevaluated. No statistical independence, overall treatment effect, new leaderboard score, or future-success extrapolation is assumed.

## Batch dependence deletion sensitivity

The869 eligible pairs originate from132 distinct CLI receipts, with rescues in47 batches. One batch contributes at most3/56 rescues. Deleting each batch and all its eligible cases gives R/E between6.1556% and6.5041%, compared with6.4442%. This deterministic deletion range is not a confidence interval and does not establish independent batches. All135 primary invocations remain in usage accounting.

## Suggested concise manuscript insertion

At the intervention query, all869 eligible pairs retain identical observation/state fingerprints, while both the language instruction and returned action hashes differ. The813 non-rescues therefore reflect failure despite a computational response to the intervention, rather than an unchanged action output; hash evidence does not establish the magnitude or physical utility of that response. Among56 rescues, the median successful continuation consumes51.67% of the remaining step budget. Cumulative confirmed rescues after25%,50%,75%, and100% of that budget are11,27,41, and56;15 finish in the last quarter. These descriptive timing results retain all869 eligible pairs in the cumulative denominator and do not test an alternative trigger.

## Recompute

Run intervention_dynamics.py with Python3 and no third-party packages. The JSON includes every eligible case, denominators, exclusions, source digests, and development-ID sensitivity. eligible_timing_cdf.csv contains all869 cases for cumulative-denominator plots (non-rescues have empty event times). rescued_timing.csv contains only56 events and must not be used to normalize the eligible-cohort CDF; its median is explicitly conditional on success.
