version=""
gpu=0
mkdir -p ./logs/${version}
export CUDA_VISIBLE_DEVICES=${gpu}
python compute_css.py \
    --json_path path/to/generated_texts.json \
    --output_path ./logs/${version}/css_scores.json