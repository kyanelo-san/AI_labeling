import torch

if torch.cuda.is_available():
    print("GPUが認識されました！")
    print(f"利用可能なGPUの数: {torch.cuda.device_count()}")
    print(f"デバイス名: {torch.cuda.get_device_name(0)}")
else:
    print("GPUが認識されませんでした。")