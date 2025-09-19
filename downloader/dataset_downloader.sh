# Run the dataset downloader
python /home/kdz/data/xzh/LLM-VirtualGuard/downloader/dataset_downloader.py \
    --source huggingface \
    --dataset sahil2801/CodeAlpaca-20k \
    --save_dir /home/kdz/data/OpenSourceDatasets/sahil2801/CodeAlpaca-20k \
    --extra use_auth_token=** \
    resume_download=True \
    endpoint="https://hf-mirror.com"

python /home/kdz/data/xzh/LLM-VirtualGuard/downloader/dataset_downloader.py \
    --source huggingface \
    --dataset nampdn-ai/tiny-codes \
    --save_dir /home/kdz/data/OpenSourceDatasets/nampdn-ai/tiny-codes \
    --extra use_auth_token=** \
    resume_download=True \
    endpoint="https://hf-mirror.com"

python /home/kdz/data/xzh/LLM-VirtualGuard/downloader/dataset_downloader.py \
    --source huggingface \
    --dataset TIGER-Lab/MathInstruct \
    --save_dir /home/kdz/data/OpenSourceDatasets/TIGER-Lab/MathInstruct \
    --extra use_auth_token=** \
    resume_download=True \
    endpoint="https://hf-mirror.com"


python /home/kdz/data/xzh/LLM-VirtualGuard/downloader/dataset_downloader.py \
    --source huggingface \
    --dataset open-r1/OpenR1-Math-220k \
    --save_dir /home/kdz/data/OpenSourceDatasets/open-r1/OpenR1-Math-220k \
    --extra use_auth_token=** \
    resume_download=True \
    endpoint="https://hf-mirror.com"


python /home/kdz/data/xzh/LLM-VirtualGuard/downloader/dataset_downloader.py \
    --source huggingface \
    --dataset lavita/ChatDoctor-HealthCareMagic-100k \
    --save_dir /home/kdz/data/OpenSourceDatasets/lavita/ChatDoctor-HealthCareMagic-100k \
    --extra use_auth_token=** \
    resume_download=True \
    endpoint="https://hf-mirror.com"


python /home/kdz/data/xzh/LLM-VirtualGuard/downloader/dataset_downloader.py \
    --source huggingface \
    --dataset Josephgflowers/Finance-Instruct-500k \
    --save_dir /home/kdz/data/OpenSourceDatasets/Josephgflowers/Finance-Instruct-500k \
    --extra use_auth_token=** \
    resume_download=True \
    endpoint="https://hf-mirror.com"

