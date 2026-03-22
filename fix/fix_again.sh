conda install -c nvidia cuda-toolkit=12.1
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

which nvcc
nvcc --version
# 确认这里显示 12.1，而不是系统 /usr/bin/nvcc 11.5
rm -rf ~/.cache/torch_extensions

python - <<'PY'
from deepspeed.ops.op_builder import FusedAdamBuilder
FusedAdamBuilder().load(verbose=True)
print("OK: fused_adam built and loadable")
PY
