# Native reset continuation v2

新包仅接续 v1 从未尝试的 968 个原始失败 case，双臂共 1936 episodes；分母仍为 1004。36 个 v1 已尝试 case、其所有错误和 response 永久保留，本版本不执行技术重试。原 counter、模型服务源码、B2000/底座、seed、horizon、半 horizon 干预、CLI 请求字段、1200 秒响应预算均不变。

`runner.episode` 从冻结 v1 复制，AST 自动验证唯一变化是 AssistanceClient 的导入路径。新 AssistanceClient 继承原 init/infer/finish，只在等待入口、每120秒及出口串行调用同一 `wire.control('rng_state')`。保存 `policy-keepalive.jsonl`，要求连接、server、seed、推理计数及四组 RNG 哈希保持相同。无后台 socket 线程、无重连、无额外 reset、无额外模型推理。任何保活异常是基础设施错误，停止新派发。

服务 idle timeout 为300秒。实际模型验证先运行原 six-client warmup，再运行以下 probe，避免占用同服务六个槽时冲突。默认全部新服务各一个 socket 并行；只做固定 processor fixture，不创建模拟 episode。每个 socket 初始 seed 一次，首个 fixture 后等待360秒并每120秒读 RNG，随后执行冻结 parity 剩余两次请求，完整动作及 RNG 都必须匹配原参考。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=$PWD:$PWD/remote_processor_candidate_v1/deps \
envs/training/bin/python -m astra_rescue_continuation_20261009.keepalive_probe \
  --services-ready results/astra-rescue-services-20261009-v2/services-ready.json \
  --output results/astra-rescue-services-20261009-v2/keepalive-probe.json
```

可用 `--service-index 0` 限定一项服务；默认360秒，不允许低于360秒。probe 不启动/清理模型，不更改服务；只写新 probe 证据。

Prepare 沿用原 runner CLI，并新增 `--prior-output`（默认 full-v1）。准备前必须取得 v1 `completion.all_children_drained=true`、外层 finished receipt、精确 owner 已退出、72 个完整不可覆盖的 claim/result。自动生成 `continuation-exclusion-ledger.json`：排除36整 case、选择968 case；不用 success 选择。`--indices` 可省略，如提供必须与自动清单顺序完全一致。

```bash
envs/sim/bin/python -m astra_rescue_continuation_20261009.runner prepare \
  --output results/astra-native-reset-continuation-20261009-v2 \
  --servers <12个已admit的servers.json路径> \
  --response-wait-seconds 1200
```

之后由外层 supervisor 使用原 `runner.environment(config)` 执行 preflight，再等待 `broker-ready.json`（model、reasoning_effort、config_sha256、remote_output 必须匹配）。队列只属于 v2。随后运行新 `runner run --config ... --parallel72 --duration-seconds28800 --cleanup-servers`。旧 drain 不改，新 output 的 drain 阻止新派发。已有 v2 claim 禁止重启/自动 resume。

仅允许一个监督器例外：可信 result 中 `initial_replay.exact_match=false` 与初态文件/参考差异一致，错误仅来自原 capture 比对门控，事件只到 reset_completed，无策略/RPC/assistance、guard 健康关闭，才派生分类 initial_state_deviation 并继续下一项。原 result 仍保留 infrastructure_unknown。其它异常、broker error、响应 timeout、invalid response 和 BrokenPipe 均停止新派发并自然排空。对已调度 case 不重采样寻找初态。

`completion.attempt_coverage_complete` 只表示所有计划 attempts 已终结；`complete` 保留原严格语义，不把 deviation 当作有效 episode。只有全覆盖且无停派原因时 run 返回0。报告新增 coverage/deviation 数，原严格四分类及 rescue 判据不变。联合1004报告应按 exclusion ledger 使用 v1的36整对和v2的968整对，永远不跨 attempt 拼配两臂。

`tests/test_continuation.py` 是本地合成 QA，无 GPU/模型/SSH。`tests/actual-v1-gate-validation.json` 是只读验证已有 v1 证据的结果：6条初态偏差接受，24条其它 infra 拒绝。真实模型保活通过与否必须以新 services 上 `keepalive-probe.json` 为准，不能把本地合成测试称为模型验证。
