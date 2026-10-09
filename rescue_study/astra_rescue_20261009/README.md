# 固定失败集的 B2000 控制重放与 Astra 条件救援

这是一套独立执行器。原始 `native-reset-b2500-20261009-v1` 的 2500 回合、1496 次成功、1004 次失败保持只读。新研究在原 1004 个失败 assignment 上做 `control` 和 `astra` 两臂，每臂原 seed、原 split、原 horizon、一次自然 reset、同一冻结 B2000 权重。仅允许 GPU 3、6 上经过身份验证的新服务；GPU 2 禁用，GPU 7 当前有其他人的任务。

这不是停滞检测器。Astra 在固定的第一个 `step >= horizon / 2`、且 `step % 16 == 0` 的查询点干预一次。等待期间不推进模拟器。动作查询仍每 16 步一次，history=4、interval=2、crop=0.95、Euler=5。Astra 子目标和原任务一起作为后半程语言输入；不修改动作模型、动作执行或成功判据。

## 当前状态

截至本地离线 QA：执行器语法检查和模拟队列检查通过，未由本包启动 GPU 服务、未运行真实 episode、未改变远端。远端出现系统关机保护，真实初始场景的逐字段一致性尚未验证。不得把离线通过写成 1004 case 已完成或真实重放已验证。

## 文件与边界

- `failure_manifest.json` / `build_failure_manifest.py`：固定原 1004 失败集及证据，由独立归档审核生成。
- `runner.py`：独立配置、manifest、自然 reset、双臂调度、原始指纹核验、结果与成对统计。
- `assistance.py`：只接收策略公开输入 `state_history, image_history, instruction`；只向 Astra 队列导出原始任务语言、当前三路 RGB 和步数预算。`state_history` 只用于转发给原动作模型和本地审计哈希，不导出给 Astra。
- `broker.py`：由本地 Windows broker 维护，读取 observation 队列并用实际 `gpt-6-astra` 返回 JSON。执行器不假定某条普通规则等价于 Astra 调用。

原始初始场景中的 XML、物理状态、layout/style、对象真值、原始结果和成功判定不会进入 Astra 请求。新 `initial_scene.json` 与原 `initial_scene` 的全部字段比较；任何差异记录到 `initial_replay.json`，并在连接/激活动作模型之前终止该 episode 为 `infrastructure_unknown`。不加载 `initial.xml` 或恢复 physics state。

## 远端依赖

项目根 `/ephemeral/qinzhen/robocasa-xr1-20261003`。将包放在根目录下 `astra_rescue_20261009/`。需原始 results、`official_b2500_v1` helpers、`native_reset_b2500_v2` 源码、冻结 B endpoint/parity、原模型和模拟器仍保持其 SHA256。

- 服务解释器：`envs/training/bin/python`；服务模块 `continuous_eval2000_v3.server`，binding 为 `results/continued-ab2000-v1/B-endpoint.json`，SHA256 `c3138b49dcf51c0514f129073870701b5cd59cb88df384ac5a4b9ac24a77c601`。
- 模拟器解释器：`envs/sim/bin/python`，torch `2.7.1+cpu`、numpy `2.2.5`、transformers `4.57.1`、mujoco `3.3.1`、robosuite `1.5.2`、robocasa `1.0.1`、gymnasium `0.29.1`，需 Pillow。
- 渲染：原私有 OSMesa，`CUDA_VISIBLE_DEVICES=''`、`PYTHONHASHSEED=0`，调用 `runner.environment(config)` 获取完整继承环境，不自行简化。
- 服务必须逐一初始化；原 campaign 对并行 CUDA 初始化的记录显示曾有 UVM 卡顿。启动由外层 launcher 负责，本执行器不抢占、清理或复用不明身份的 GPU 进程。

## 准备和运行

以下命令只是入口说明，启动前应完成当前 GPU owner/health 审计、服务身份和实际并发前向/RNG parity 检查。不能复用已结束 campaign 的陈旧 owner manifests。

```text
envs/training/bin/python -m astra_rescue_20261009.runner prepare \
  --output results/astra-native-reset-pilot-20261009-v1 \
  --failure-manifest astra_rescue_20261009/failure_manifest.json \
  --servers <new-service-1/servers.json> <new-service-2/servers.json> \
  --indices <original-assignment-index-1> <index-2> <index-3> <index-4> \
  --arms control astra --response-wait-seconds 1200
```

`--indices` 仅接受原失败集中的 assignment index；不提供即使用全部 1004。原 job 字段保留，新增 `original_id` 和 `rescue_arm`，episode ID 后缀为 `--control` 或 `--astra`。双臂全量计划 2008 回合，不是 1004 回合。每个新 output 必须是项目 `results/` 下的直接子目录，名字必须包含 `astra` 和 `native-reset`。

通过 Python 外层驱动读取新 config，用 `env=runner.environment(config)` 启动：

```text
envs/sim/bin/python -m astra_rescue_20261009.runner preflight --config <new-output/config.json>
envs/sim/bin/python -m astra_rescue_20261009.runner run --config <new-output/config.json> \
  --parallel 72 --duration-seconds <explicit-wall-budget>
```

并发受 `min(--parallel, 6 * 服务数)` 限制。`run` 在任何 simulator `infrastructure_unknown` 后停止新派发并自然排空已有 worker。Astra queue 超时/错误继续原语言输入，单独标记 assistance 失败，不伪装成已接受的 Astra 计划。

本地 broker 的 `--once` 会在完整 `--run-seconds` 预算内等待首个请求；`--batch-wait-seconds` 只从首个请求到达后开始凑批计时。CLI 额度、认证、执行等错误会保留本地 `stop.json`，并只尝试一次向同一远端 output 原子、exclusive 写入 `drain.request.json`；写入前核验该目录 `config.json` 的研究 schema 和精确 output 路径。runner 随后停止新派发，已有 worker 自然排空（仍受原定硬截止预算约束）；broker 不发送进程信号、不覆盖已有 marker。远端发送失败单独记录，不掩盖原 CLI 错误或无限重试。服务恢复或额度重置后，resume 前须人工检查本地 stop 与远端 drain marker；程序不会自动删除它们。

不要直接使用默认 13500 秒作为全量双臂预算。原 1004 失败回合累计 worker wall time 为 785051 秒，均值 781.9 秒、P90 1347.5 秒；72 并发的单臂理想下界约 3.03 小时，双臂约 6.06 小时，另需初始化、任务尾部和 Astra 等待。真实吞吐必须由 pilot 再估计。

## 队列契约

目录：`<new-output>/requests/<original-job-id>--astra/`。执行器先写 `left.png/right.png/wrist.png`，再原子写 `request.json` 作为 ready 信号。三个图片都是当前查询的最新帧，无附加渲染、缩放或截图。

```json
{
  "schema": "astra_observation_request_v1",
  "request_id": "native-B-0339-TurnOffStove-2090900339--astra",
  "request_token": "<32 lowercase hex characters from operating-system entropy>",
  "case_id": "native-B-0339-TurnOffStove-2090900339",
  "step": 384,
  "horizon": 750,
  "remaining_steps": 366,
  "original_instruction": "Turn off the front right burner of the stove.",
  "cameras": [
    {"name":"left","file":"left.png","sha256":"<actual-png-sha256>"},
    {"name":"right","file":"right.png","sha256":"<actual-png-sha256>"},
    {"name":"wrist","file":"wrist.png","sha256":"<actual-png-sha256>"}
  ],
  "permitted_inputs": ["original_instruction","current_rgb","step_budget"],
  "oracle_inputs_included": false
}
```

broker 先在同目录发布 `cli_receipt.json`、`cli_stdout.jsonl`、`cli_final_output.json`，再原子发布 `response.json`，不得覆盖任何既有证据。要求 request ID、request token、request 文件的实际 SHA256、模型名完全相符。token 来自系统熵，不改变 Python/NumPy/Torch 的策略随机流。`status=ok` 的 `subgoal_instruction` 必须是去首尾空白后 1–1500 字符的字符串。`status=error` 保留错误与调用证据。响应必须附带 `cli_receipt_file` 和真实回执的 `cli_receipt_sha256` 才能通过严格救回审核。

```json
{
  "schema":"astra_observation_response_v1",
  "request_id":"native-B-0339-TurnOffStove-2090900339--astra",
  "request_token":"<exact request token>",
  "request_sha256":"<exact-request-file-sha256>",
  "model":"gpt-6-astra",
  "status":"ok",
  "subgoal_instruction":"<actual Astra output>",
  "batch_id":"<broker batch ID>",
  "cli_receipt_file":"cli_receipt.json",
  "cli_receipt_sha256":"<actual invocation receipt SHA256>"
}
```

`status` 可为 `applied/control/not_reached/broker_error/invalid_response/timeout`。不匹配、JSON 错误和超时不会再请求一次。`assistance.json` 保留等待时长与最终 policy instruction。总 horizon 不因等待增加。

## QA 和报告

本地 `probe/queue-qa-bjf8zinq/receipt.json` 检查了 control 不发请求、固定中点一次成功响应、broker error、错误 digest、timeout 五条路径；并验证 PNG 解码后与原当前 RGB 字节一致、干预前输入不变、四次策略查询均留 trace。

真实 pilot 的通过标准还包括：自然 reset 全部原始指纹 exact-match、每臂 horizon/step/RNG query 数与原协议一致、模型 hello 身份仍是冻结 B2000、readback guard 健康、control/Astra 干预前 query/action trace 完全一致。真实场景重放尚未执行，因此这些条件不能用离线 QA 代替。

`analysis/report.json` 保留原始双臂成功数和四格成对结果，但 `paired.astra_only` 只是原始结果，**不能称作救回**。`strict_rescued` 必须同时满足：控制臂失败、Astra 臂成功、确实应用干预、两臂原始初始指纹匹配、该 case 干预前 trace 完全一致、请求/响应哈希和 token 一致，以及实际 Codex CLI 调用回执、完成事件和原始输出均可验证。

回执的 schema 为 `astra_cli_receipt_v1`，至少包括 `cli_executed=true`、`model/model_requested=gpt-6-astra`、`returncode=0`、`status=ok`、`no_tool_calls=true`、`output_validated=true`、实际 `command` 数组、`requests=[{request_id,request_sha256,request_token}]`，以及 `stdout_file/stdout_sha256` 和 `raw_answer_file/raw_answer_sha256`。执行器检查实际命令为 Codex、命令中的 `--model` 值、JSON 事件中不存在工具调用/失败、存在含非零 token usage 的 `turn.completed`，并且 `agent_message.text` 的 JSON 与 raw output 一致且包含本 case 的 token 和子目标；再用策略 query trace 验证该指令确实被传给 B2000。只填一个 `model` 字符串无法通过审核。这是实际 CLI 调用记录的审计，不是服务方签名的模型身份证明。

`analysis/case-classification.json` 始终列出固定的全部 1004 case，每个 case 的分类互斥：`rescued`、`not_rescued`、`unknown`、`initial_state_deviation`。错误、超时、缺失/未派发 episode、无法验证的调用证据和 prefix 偏差保留为 unknown；初始指纹明确不符单独列为 deviation。未应用干预的自然重试成功不计救回。pilot 未选的 case 仍列为 unknown，并标注 `not_selected_in_this_run`。

报告同时给固定分母 1004 和本次 selected/pilot 分母，绝不删除 pending/unknown。尚有 unknown 或 deviation 时，`strict_rescue_rate_selected=null`；已确认 strict rescued 除以固定分母/selected 分母只作为保守的已确认比例。即便执行中止，也走同一分类汇总。`analysis/prefix-parity.json` 逐 case 保存干预前 input/action 检查。

`probe/strict-analysis-qa-3_7ipg77/qa_receipt.json` 是隔离的合成测试，15 条路径验证只有完整有效链计为 strict rescued；broker error、timeout、无干预成功、prefix 偏差、初始偏差、缺结果、token/hash/model/CLI 输出失配均不会误计。合成测试未执行模拟器或模型，其数字不是复测结果。

原 1496 成功回合未重测，因此不能报告新的全 2500 基准分数或证明无回归。默认运行 deadline 未修改；外层 root 必须依据至少约 6 小时的双臂理想下界，加加载、尾部和 Astra 等待后显式指定预算。
