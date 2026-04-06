from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="laion/laion2B-en",
    filename="laion2B-en-100k.parquet",
    repo_type="dataset",
    local_dir=r"C:\v9\datasets"
)

print("Downloaded to:", path)