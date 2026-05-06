from chandra.model import InferenceManager
from chandra.model.schema import BatchInputItem
from PIL import Image

# Start vLLM server first with: chandra_vllm
manager = InferenceManager(method="vllm")
batch = [
    BatchInputItem(
        image=Image.open("/root/tungn197/idp/data/test_samples/bol1.png"),
        prompt_type="ocr_layout"
    )
]
result = manager.generate(batch)[0]
print(result.markdown)