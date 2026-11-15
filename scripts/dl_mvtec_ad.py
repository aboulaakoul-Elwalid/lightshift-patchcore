from huggingface_hub import hf_hub_download
p = hf_hub_download(
    "hdtech/mvtech_anomaly_detection",
    "mvtech_anomaly_detection.zip",
    repo_type="dataset",
    local_dir="/tmp/mvtec_dl",
)
print("DONE", p)
