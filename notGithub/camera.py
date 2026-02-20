import cv2
import torch
import numpy as np
from ultralytics import YOLO

def main():
    # GPUが利用可能か確認
    if not torch.cuda.is_available():
        print("GPUが利用できません。CPUを使用して検出します。")
        device = 'cpu'
    else:
        print("GPUが利用可能です。GPUを使用して検出します。")
        device = 'cuda'

    model = YOLO('D:/dataset\/runs/detect/train23/weights/best.pt')

    # カメラの準備
    cap = cv2.VideoCapture(0)
    
    # カメラの解像度を設定
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # カメラが正しく開かれたか確認
    if not cap.isOpened():
        print("カメラを開けませんでした。")
        return

    # ループ処理でリアルタイム検出
    while True:
        # フレームをキャプチャ
        ret, frame = cap.read()
        if not ret:
            break

        # フレームに物体検出を実行（追跡も行う）
        results = model.track(frame, persist=True, device=device)

        # 検出結果の取得
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        
        # オリジナルのフレームにカスタムで描画
        annotated_frame = frame.copy()
        
        # カスタム描画関数でバウンディングボックスを描画
        annotated_frame = draw_custom_boxes(annotated_frame, boxes, class_ids, model.names)

        # 描画されたフレームを表示
        cv2.imshow("Real-time Detection", annotated_frame)

        # 'q'キーが押されたらループを終了
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # キャプチャを解放し、ウィンドウを閉じる
    cap.release()
    cv2.destroyAllWindows()

# カスタム描画関数
def draw_custom_boxes(image, boxes, class_ids, class_names):
    # クラスごとに色を設定
    # ここで色（BGR形式）をカスタマイズできます
    colors = {
        0: (0, 0, 255),    # クラスID 0: 赤
        1: (0, 255, 0),    # クラスID 1: 緑
        2: (255, 0, 0),    # クラスID 2: 青
        # 他のクラスIDと色を追加
    }

    # フォントの設定
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    line_thickness = 2

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        class_id = class_ids[i]
        label = class_names[class_id]
        color = colors.get(class_id, (0, 255, 255)) # クラスIDに対応する色を取得、なければ黄色

        # バウンディングボックスの描画
        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)

        # ラベルのテキストサイズを計算
        (label_width, label_height), _ = cv2.getTextSize(label, font, font_scale, line_thickness)
        
        # ラベルの背景の描画
        cv2.rectangle(image, (x1, y1 - label_height - 5), (x1 + label_width + 5, y1), color, -1)

        # ラベルのテキストの描画
        cv2.putText(image, label, (x1 + 2, y1 - 2), font, font_scale, (255, 255, 255), line_thickness)
    
    return image

if __name__ == '__main__':
    main()
