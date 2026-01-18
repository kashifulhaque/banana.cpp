import os
import struct
from transformers import GPT2LMHeadModel

os.makedirs("weights", exist_ok=True)

### load the smallest GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

### get the raw data in float32
state_dict = model.state_dict()

with open("./weights/gpt2_weights.bin", "wb") as file:
    for name, param in state_dict.items():
        name_bytes = name.encode("utf-8")
        file.write(struct.pack("i", len(name_bytes)))
        file.write(name_bytes)

        shape = param.shape
        file.write(struct.pack("i", len(shape)))
        for dim in shape:
            file.write(struct.pack("i", dim))

        data = param.detach().cpu().numpy().astype("float32")
        file.write(data.tobytes())

print(f"Successfully exported {len(state_dict)} tensors to gpt2_weights.bin")
