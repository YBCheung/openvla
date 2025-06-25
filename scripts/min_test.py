# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import time

import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
img_path = "../datasets/beignets-task-guide.png"
ckpt_path = "./runs/openvla-7b+spot_kitchen+b16+lr-2e-05+lora-r32+dropout-0.0--image_aug--10000_chkpt"
hf_path = "openvla/openvla-7b"
# Load Processor & VLA
processor = AutoProcessor.from_pretrained(ckpt_path, trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    ckpt_path,
    attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
    trust_remote_code=True
).to(device)

count = 0
while(count < 10):
    
    start_time = time.time()
    # Grab image input & format prompt
    image = Image.open(img_path)

    prompt = "Pick up the blue stick, lift it up, then put it back"

    # Predict Action (7-DoF; un-normalize for BridgeV2)
    inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
    action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    
    freq = 1 / (time.time() - start_time)
    print(action, freq)
    count += 1

# Execute...
# robot.act(action, ...)
