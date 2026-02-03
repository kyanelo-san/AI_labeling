from ultralytics import YOLO
import torch

def main():
    if not torch.cuda.is_available():
        print("GPUが利用できません。CPUを使用して学習します。")
        device = 'cpu'
    else:
        print("GPUが利用可能です。GPUを使用して学習します。")
        device = 'cuda'
        
    torch.cuda.empty_cache()

    model = YOLO('yolo11n.pt')
    results = model.train(data='data.yaml', epochs=100, imgsz=1280, device=device, batch=12, workers=0)

if __name__ == '__main__':
    main()
