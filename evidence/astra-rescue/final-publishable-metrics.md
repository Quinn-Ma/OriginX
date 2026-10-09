# Astra rescue study: audited metrics for publication

Publication copy of the final independent audit summary; machine-readable evidence remains unchanged.

Across the fixed subset of 1,004 original failures, the claim-frozen v1/continuation partition contains each case exactly once (36 + 968). All 2,008 planned control/Astra attempts have terminal records. Strict classification yields **56 rescued, 813 not rescued, 28 unknown, and 107 initial-state deviations**. The observed strict rescue proportion is **56/1,004 = 5.58%**; all unknown and deviation cases remain in this denominator.

There are **M = 874** matched completed replay pairs and **E = 869** matched pairs with successfully applied Astra assistance and valid evidence. M includes five technical-unknown pairs without verified Astra application. The conditional ratios are **56/874 = 6.41%** and **56/869 = 6.44%**. Raw paired outcomes are 56 Astra-only successes, 0 control-only successes, 0 both-successes and 818 both-failures; raw outcomes are separate from strict rescue classification.

The original benchmark remains **1,496/2,500 (59.84%)**. These are conditional failure-subset study results and do not replace the official benchmark score. Attempt coverage is complete; outcome completeness is false because unknown and initial-state-deviation cases remain.

| Stratum | Fixed cases | Rescued | Not rescued | Unknown | Initial deviation | M | E | R / fixed cases | R / E |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| composite_seen | 321 | 25 | 218 | 8 | 70 | 244 | 243 | 7.79% | 10.29% |
| composite_unseen | 530 | 19 | 470 | 20 | 21 | 493 | 489 | 3.58% | 3.89% |
| atomic_seen | 153 | 12 | 125 | 0 | 16 | 137 | 137 | 7.84% | 8.76% |

27 of 49 tasks contain at least one strict rescue. The following tasks have the highest rescue counts (counts do not imply a controlled comparison across task difficulty):

| Task | Rescued / original failures | M | E |
|---|---:|---:|---:|
| BreadSelection | 7 / 30 | 29 | 29 |
| PrepareCoffee | 6 / 39 | 39 | 39 |
| CuttingToolSelection | 4 / 26 | 26 | 26 |
| KettleBoiling | 4 / 17 | 14 | 14 |
| CloseToasterOvenDoor | 3 / 6 | 6 | 6 |
| LoadDishwasher | 3 / 13 | 13 | 13 |
| SetUpCuttingStation | 3 / 12 | 12 | 12 |
| StirVegetables | 3 / 23 | 17 | 17 |
| PackIdenticalLunches | 2 / 32 | 31 | 31 |
| RecycleBottlesByType | 2 / 38 | 37 | 37 |

Usage is deduplicated by unique CLI receipt SHA / batch identity, never multiplied by the number of requests in a batch. Primary usage comprises **135 CLI calls (134 successful, one quota error)**; the development pilot adds **one separate CLI call**. All invocations, including errors and unused responses, remain counted.

**Primary known token counts:** input 2,592,713; output 195,237; reasoning 75,511; cached input 0. One quota-error call has no token counters, so these are known sums/lower bounds and complete totals remain null. Reasoning and output counters are reported separately and not added together. Dollar cost is **null**, with no API price inferred from authenticated CLI quota.

**Primary CLI latency:** sum 6186.438 seconds across 135 calls; mean 45.825 s; median 46.157 s; range 2.334–77.351 s. These are per-call latencies, not rollout durations or per-request latency. Pilot input/output/reasoning tokens are 19,071 / 629 / 137 and its single-call latency is 24.349 s.

Terminal episode accounting: {'infrastructure_unknown': 171, 'completed': 1837}. Case-level initial deviations and episode-level infrastructure-unknown counts use different units and must not be interchanged.

Independent read-only process audit verified all **1,951** recorded v2 Linux identities absent from /proc: 1,936 workers, 12 model services and three runner/campaign/launcher owners. No signals or process modifications were used; this does not assert availability of GPUs occupied by other jobs.

Archive verification: v1 1,014 files; continuation 29,177 files. Archive SHA/size and every extracted file SHA were independently checked.

Audit SHA256: `fb051322aa7bc26130f908bcd7635ec0aff2f9c00875561b5d3d9ad8252901e9`.
Cleanup receipt SHA256: `4bdaf0289384805bea2e0090e97713e3e6022bbc2deb63840b66c4a87b7174b4`.
