#!/usr/bin/env bash
# Build the three pinned Python envs for the mask study on the HKUST cluster.
# Envs live on the RAM tier (/dev/shm/qinzhen, regenerable from the lock files kept on disk).
set -uo pipefail
T=/ephemeral/qinzhen/originx-harness-v2
W=$T/work/hkust
E=/dev/shm/qinzhen/originx-envs
export UV_CACHE_DIR=/dev/shm/qinzhen/.uvcache UV_LINK_MODE=copy
LOG=$W/build_envs.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
for i in $(seq 1 120); do
  [ -f $W/planner-cloud.requirements.txt ] && [ -f $W/policy-cloud.requirements.txt ] && [ -f $W/autodl-sim-requirements.lock ] \
  && [ -f $T/work/id60-masks-validation/deps/wheels/supervision-0.27.0-py3-none-any.whl ] && break; sleep 5; done
mkdir -p $E $UV_CACHE_DIR
PY=/usr/bin/python3.12; UV=/snap/bin/uv
log "BUILD_START python=$($PY -V 2>&1) uv=$($UV --version)"
# --- sim env (pinned lock, binary only, no deps) + supervision/defusedxml wheels
[ -x $E/sim/bin/python ] || $UV venv --python $PY $E/sim >>"$LOG" 2>&1
$UV pip install --python $E/sim/bin/python --no-deps --only-binary :all: -r $W/autodl-sim-requirements.lock >>"$LOG" 2>&1 \
 && $UV pip install --python $E/sim/bin/python --no-deps --no-index $T/work/id60-masks-validation/deps/wheels/*.whl >>"$LOG" 2>&1 \
 && $E/sim/bin/python -c "import mujoco,supervision,numpy,cv2,defusedxml,scipy;print('SIM_OK',mujoco.__version__,supervision.__version__,numpy.__version__)" >>"$LOG" 2>&1 \
 && log "ENV_DONE sim" || log "ENV_FAIL sim"
$UV pip freeze --python $E/sim/bin/python > $W/sim-installed-freeze.txt 2>/dev/null
# --- policy / planner envs: torch 2.8.0+cu128 + torchvision 0.23.0+cu128 (the AutoDL base), then the frozen lists
for role in policy planner; do
  [ -x $E/$role/bin/python ] || $UV venv --python $PY $E/$role >>"$LOG" 2>&1
  $UV pip install --python $E/$role/bin/python "torch==2.8.0+cu128" "torchvision==0.23.0+cu128" \
     --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match >>"$LOG" 2>&1 \
     || { log "ENV_FAIL $role torch"; continue; }
  $UV pip install --python $E/$role/bin/python -r $W/$role-cloud.requirements.txt >>"$LOG" 2>&1 || { log "ENV_FAIL $role requirements"; continue; }
  log "ENV_DONE $role"
done
WHL=/dev/shm/qinzhen/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
[ -f "$WHL" ] || curl -L -sS -o "$WHL" "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" >>"$LOG" 2>&1
echo "f25da18657a87fc83dc1bfb8b7751b82246e9db355510226b674fd437c34b5fb  $WHL" | sha256sum -c >>"$LOG" 2>&1 \
 && $UV pip install --python $E/policy/bin/python --no-deps "$WHL" >>"$LOG" 2>&1 && log "FLASH_OK" || log "FLASH_FAIL"
$E/policy/bin/python -c "import torch,flash_attn,transformers,numpy,mmengine;print('POLICY_OK',torch.__version__,flash_attn.__version__,transformers.__version__,numpy.__version__)" >>"$LOG" 2>&1 || log "POLICY_IMPORT_FAIL"
$E/planner/bin/python -c "import torch,transformers,numpy,diffusers;print('PLANNER_OK',torch.__version__,transformers.__version__,numpy.__version__)" >>"$LOG" 2>&1 || log "PLANNER_IMPORT_FAIL"
for role in policy planner; do
  $UV pip freeze --python $E/$role/bin/python > $W/$role-installed-freeze.txt 2>/dev/null
  echo "== uv pip check $role" >>"$LOG"; $UV pip check --python $E/$role/bin/python >>"$LOG" 2>&1
done
du -sh $E/* >>"$LOG" 2>&1
log "ALL_ENVS_DONE"
