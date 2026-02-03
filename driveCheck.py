import cv2
import torch
import numpy as np
from ultralytics import YOLO
import os
import io
import time
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ================= 設定部分 =================
# Google Drive設定
SERVICE_ACCOUNT_FILE = 'credentials.json'
FOLDER_ID = '1P0kb5GhoYKZKv0x8BOhFnOISLwFO-D2c'

# ディレクトリ設定
DOWNLOAD_DIR = 'downloaded_images'
OUTPUT_DIR = 'static'
DEMO_OUTPUT_DIR = 'demo/static'

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DEMO_OUTPUT_DIR, exist_ok=True)

# ★モデル設定（2つ使います）
# 1. 自販機解析用の自作モデル
VENDING_MODEL_PATH = 'D:/dataset/runs/detect/train23/weights/best.pt'
# 2. 人検知用の標準モデル（初回実行時に自動ダウンロードされます）
PERSON_MODEL_NAME = 'yolov8n.pt' 
# ===========================================

def download_latest_image():
    """Driveから最新の画像をダウンロード"""
    try:
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=['https://www.googleapis.com/auth/drive'])
        service = build('drive', 'v3', credentials=creds)

        results = service.files().list(
            q=f"'{FOLDER_ID}' in parents and mimeType contains 'image/' and trashed=false",
            orderBy='createdTime desc',
            pageSize=1,
            fields="files(id, name)"
        ).execute()

        items = results.get('files', [])
        if not items:
            return None, None

        latest_file = items[0]
        file_id = latest_file['id']
        file_name = latest_file['name']
        
        save_path = os.path.join(DOWNLOAD_DIR, file_name)

        if os.path.exists(save_path):
            return save_path, file_name

        print(f"新しい画像をダウンロード中: {file_name}")
        request = service.files().get_media(fileId=file_id)
        with io.FileIO(save_path, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
        
        return save_path, file_name

    except Exception as e:
        print(f"Drive API Error: {e}")
        return None, None

def draw_custom_boxes(image, boxes, class_ids, class_names):
    """描画関数"""
    colors = {0: (0, 0, 255), 1: (0, 255, 0), 2: (255, 0, 0)}
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    line_thickness = 2

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        class_id = class_ids[i]
        label = class_names[class_id]
        color = colors.get(class_id, (0, 255, 255))

        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)
        (label_width, label_height), _ = cv2.getTextSize(label, font, font_scale, line_thickness)
        cv2.rectangle(image, (x1, y1 - label_height - 5), (x1 + label_width + 5, y1), color, -1)
        cv2.putText(image, label, (x1 + 2, y1 - 2), font, font_scale, (255, 255, 255), line_thickness)
    
    return image

def main():
    # ---------------------------------------------------------
    # GPU設定
    # ---------------------------------------------------------
    if torch.cuda.is_available():
        print("★ GPU(CUDA)を使用して検出します。")
        device = 'cuda'
        torch.cuda.empty_cache()
    else:
        print("GPUが見つかりません。CPUを使用します。")
        device = 'cpu'

    # モデル読み込み
    print(f"自販機モデル読み込み中: {VENDING_MODEL_PATH}")
    try:
        model_vending = YOLO(VENDING_MODEL_PATH)
        print("人検知モデル読み込み中(標準YOLO)...")
        model_person = YOLO(PERSON_MODEL_NAME) # 人検知専用
    except Exception as e:
        print(f"モデルの読み込みに失敗しました: {e}")
        return

    print("監視を開始します。(Ctrl+Cで終了)")
    last_processed_file = None

    while True:
        image_path, file_name = download_latest_image()

        if image_path and file_name != last_processed_file:
            print(f"チェック開始: {file_name}")
            
            frame = cv2.imread(image_path)
            if frame is None:
                print("画像の読み込みに失敗しました。")
                continue

            height, width = frame.shape[:2]

            # ========================================================
            # ステップ1: 人がいるかチェック (標準モデルを使用)
            # classes=[0] は「人(person)」クラスのみを検出する設定です
            # ========================================================
            person_results = model_person.predict(frame, classes=[0], conf=0.5, device=device, verbose=False)
            
            # 人が検出された場合（boxesの数が0より多い）
            if len(person_results[0].boxes) > 0:
                print(f"★ 警告: 画像に人が検出されました。プライバシー保護のためスキップします。")
                
                # スキップ処理:
                # current.jpgを更新せず、次のループへ行きます。
                # Webサイト側は古い画像（人がいない画像）を表示し続けます。
                last_processed_file = file_name
                time.sleep(10)
                continue # ここでループの先頭に戻る

            # ========================================================
            # ステップ2: 人がいなければ自販機解析を実行
            # ========================================================
            print("人は検出されませんでした。自販機解析を実行します。")
            results = model_vending.predict(frame, device=device)

            if results and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                
                annotated_frame = frame.copy()
                annotated_frame = draw_custom_boxes(annotated_frame, boxes, class_ids, model_vending.names)

                history_path = os.path.join(OUTPUT_DIR, f"result_{file_name}")
                cv2.imwrite(history_path, annotated_frame)
                
                current_path = os.path.join(OUTPUT_DIR, "current.jpg")
                cv2.imwrite(current_path, annotated_frame)
                
                print(f"ウェブサイト画像を更新しました: {current_path}")
            else:
                print("自販機の検出対象が見つかりませんでした(画像は更新しません)。")

            last_processed_file = file_name
        
        time.sleep(10) 

if __name__ == '__main__':
    main()