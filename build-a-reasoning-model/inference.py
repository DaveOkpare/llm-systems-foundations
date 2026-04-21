from pathlib import Path

import torch
from reasoning_from_scratch.qwen3 import QWEN_CONFIG_06_B, Qwen3Model, Qwen3Tokenizer


def get_device(enable_tensor_cores=True):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA CUDA GPU")

        if enable_tensor_cores:
            major, minor = map(int, torch.__version__.split(".")[:2])
            if (major, minor) >= (2, 9):
                torch.backends.cuda.matmul.fp32_precision = "tf32"
                torch.backends.cudnn.conv.fp32_precision = "tf32"
            else:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")

    elif torch.xpu.is_available():
        device = torch.device("xpu")
        print("Using Intel GPU")

    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device


device = torch.device("cpu")  # get_device()

tokenizer_path = Path("qwen3") / "tokenizer-base.json"
tokenizer = Qwen3Tokenizer(str(tokenizer_path))

model_path = Path("qwen3") / "qwen3-0.6B-base.pth"
model = Qwen3Model(QWEN_CONFIG_06_B)
model.load_state_dict(torch.load(model_path))
model.to(device)

if __name__ == "__main__":
    prompt = "Explain large language models."
    input_token_ids_list = tokenizer.encode(prompt)
    print(f"Number of input tokens: {len(input_token_ids_list)}")

    input_tensor = torch.tensor(input_token_ids_list)
    input_tensor_fmt = input_tensor.unsqueeze(0).to(device)

    with torch.inference_mode():
        output_tensor = model(input_tensor_fmt)

    output_tensor_fmt = output_tensor.squeeze(0)
    # decoded = tokenizer.decode(output_tensor_fmt)
    # print(decoded)
    print(f"Formatted Output tensor shape: {output_tensor_fmt.shape}")
    last_token_id = output_tensor_fmt[-1].argmax(dim=-1, keepdim=True)
    last_token = tokenizer.decode(last_token_id)
    print(f"Last token: {last_token}")
