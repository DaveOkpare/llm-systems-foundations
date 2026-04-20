from reasoning_from_scratch.qwen3 import download_qwen3_small

if __name__ == "__main__":
    download_qwen3_small(kind="base", tokenizer_only=True, out_dir="qwen3")
    download_qwen3_small(kind="base", tokenizer_only=False, out_dir="qwen3")
