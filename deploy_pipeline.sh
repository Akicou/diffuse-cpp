#!/bin/bash
set -e

export HF_TOKEN="${HF_TOKEN}"
MODEL_DIR="/workspace/LLaDA2.2-flash"
REPO_DIR="/workspace/diffuse-cpp"
F16_OUT="/workspace/LLaDA2.2-flash-F16.gguf"
Q4_OUT="/workspace/LLaDA2.2-flash-Q4_K_S.gguf"

echo "============================================"
echo "  LLaDA2.2-flash GGUF Pipeline"
echo "============================================"
echo "Start: $(date)"

# Step 1: Convert to F16 GGUF
echo ""
echo "[1/3] Converting to F16 GGUF..."
python3 /workspace/convert-llada2-fast.py \
    --input "$MODEL_DIR" \
    --output "$F16_OUT" \
    --type f16

F16_SIZE=$(du -h "$F16_OUT" | cut -f1)
echo "F16 GGUF: $F16_SIZE"

# Step 2: Quantize to Q4_K_S
echo ""
echo "[2/3] Quantizing to Q4_K_S..."
cd "$REPO_DIR"
export PATH=/usr/local/cuda/bin:$PATH
./build/diffuse-quantize "$F16_OUT" "$Q4_OUT" Q4_K_S

Q4_SIZE=$(du -h "$Q4_OUT" | cut -f1)
echo "Q4_K_S GGUF: $Q4_SIZE"

# Step 3: Create HF repo and upload
echo ""
echo "[3/3] Uploading to HuggingFace..."
python3 -c "
from huggingface_hub import HfApi, create_repo
api = HfApi(token='$HF_TOKEN')

# Create repo
try:
    create_repo('Akicou/inclusionAI_LLaDA2.2-flash-GGUF', repo_type='model', token='$HF_TOKEN', exist_ok=True)
except:
    pass

# Upload README
api.upload_file(
    path_or_fileobj='/workspace/hf_README.md',
    path_in_repo='README.md',
    repo_id='Akicou/inclusionAI_LLaDA2.2-flash-GGUF',
    token='$HF_TOKEN',
)
print('README uploaded')

# Upload F16
print(f'Uploading F16 ($F16_SIZE)...')
api.upload_file(
    path_or_fileobj='$F16_OUT',
    path_in_repo='LLaDA2.2-flash-F16.gguf',
    repo_id='Akicou/inclusionAI_LLaDA2.2-flash-GGUF',
    token='$HF_TOKEN',
)
print('F16 uploaded')

# Upload Q4_K_S
print(f'Uploading Q4_K_S ($Q4_SIZE)...')
api.upload_file(
    path_or_fileobj='$Q4_OUT',
    path_in_repo='LLaDA2.2-flash-Q4_K_S.gguf',
    repo_id='Akicou/inclusionAI_LLaDA2.2-flash-GGUF',
    token='$HF_TOKEN',
)
print('Q4_K_S uploaded')
print('All done!')
"

echo ""
echo "============================================"
echo "  Pipeline complete: $(date)"
echo "============================================"
