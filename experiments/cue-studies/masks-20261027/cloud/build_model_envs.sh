set -euo pipefail
T=/root/autodl-tmp/originx-harness-v2/work/autodl
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
for role in policy planner; do
  E=/root/originx-$role-env
  [ -x $E/bin/python ] || /root/miniconda3/bin/python -m venv --system-site-packages $E
  $E/bin/python -m pip install --no-cache-dir --index-url https://mirrors.aliyun.com/pypi/simple -r $T/$role-cloud.requirements.txt
  echo "ENV_DONE $role"
done
source /etc/network_turbo
W=/root/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
[ -f $W ] || curl -L -o $W "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
echo "f25da18657a87fc83dc1bfb8b7751b82246e9db355510226b674fd437c34b5fb  $W" | sha256sum -c
/root/originx-policy-env/bin/python -m pip install --no-deps $W
/root/originx-policy-env/bin/python -c "import torch,flash_attn,transformers;print('POLICY_OK',torch.__version__,flash_attn.__version__,transformers.__version__)"
/root/originx-planner-env/bin/python -c "import torch,transformers;print('PLANNER_OK',torch.__version__,transformers.__version__)"
echo ALL_ENVS_DONE
