# ```Examples
# python model_downloader.py \
#     --source huggingface \
#     --model mistralai/Mistral-7B-v0.3 \
#     --save_dir /home/kdz/data/OpenSourceModels/mistralai/Mistral-7B-v0.3 \
#     --extra use_auth_token=** \
#     resume_download=True \
#     ignore_patterns='["*original*"]' \
#     endpoint="https://hf-mirror.com" \
#     force_download=True
# ```

python model_downloader.py \
    --source huggingface \
    --model mistralai/Mistral-7B-v0.3 \
    --save_dir /home/kdz/data/OpenSourceModels/mistralai/Mistral-7B-v0.3 \
    --extra use_auth_token=** \
    resume_download=True \
    endpoint="https://hf-mirror.com"

python model_downloader.py \
    --source huggingface \
    --model meta-llama/Llama-3.1-8B \
    --save_dir /home/kdz/data/OpenSourceModels/meta-llama/Llama-3.1-8B \
    --extra use_auth_token=** \
    resume_download=True \
    ignore_patterns='["*original*"]' \
    endpoint="https://hf-mirror.com"