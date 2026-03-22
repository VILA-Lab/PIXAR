# 1) Upgrade to PyTorch's CUDA 12.x release (example: cu121)
pip install --no-cache-dir --upgrade \
  torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2) Reinstall bitsandbytes (newer versions support cu12x)
pip install --no-cache-dir --upgrade bitsandbytes

# 3) Verify version alignment
python - <<'PY'
import torch, os
print("torch:", torch.__version__, "torch.cuda:", torch.version.cuda)
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH",""))
PY

# inside your pixar env
pip show torch deepspeed | sed 's/^/>>> /'

# upgrade DS to a compatible version
pip install -U "deepspeed>=0.14.4"

# verify
python - <<'PY'
import torch, deepspeed
print("torch:", torch.__version__)
print("deepspeed:", deepspeed.__version__)
PY

pip install -U hf_transfer


# In the pixar env
conda install -y -c conda-forge "libstdcxx-ng>=12.2" "gcc=12.*" "gxx=12.*"

# Make sure the env's lib dir is searched first
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"

# Clean previously built (bad) extensions so they recompile
rm -rf ~/.cache/torch_extensions/*

# Quick check that the symbol exists in the conda libstdc++
strings "$CONDA_PREFIX/lib/libstdc++.so.6" | grep GLIBCXX_3.4.32 || echo "Not found"

# Which libstdc++ will Python actually load first?
python - <<'PY'
import os, sys
print("CONDA_PREFIX:", os.environ.get("CONDA_PREFIX"))
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))
PY

# Can we import FusedAdam after fixing?
python - <<'PY'
from deepspeed.ops.adam import FusedAdam
print("FusedAdam import OK")
PY