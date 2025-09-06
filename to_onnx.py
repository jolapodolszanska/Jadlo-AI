import torch
from model_torch import FoodClassifier
dummy_input = torch.randn(1, 3, 224, 224)
model = FoodClassifier(num_classes=101)
model.load_state_dict(torch.load("food101_model.pt", map_location="cpu"))
model.eval()

torch.onnx.export(
    model,
    dummy_input,
    "food101.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)
print("✅ Wyeksportowano food101.onnx")
