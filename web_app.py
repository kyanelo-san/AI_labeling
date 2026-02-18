from flask import Flask, render_template
import cv2
import threading
import time
from datetime import datetime
from ultralytics import YOLO
import os
import io

# --- Google Drive API用ライブラリ ---
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

app = Flask(__name__)

# ===========================
# 【重要】設定項目：ここを書き換えてください
# ===========================
# 1. YOLOモデルのパス
MODEL_PATH = 'D:/dataset/runs/detect/train30/weights/best.pt'
# 2. ダウンロードしたサービスアカウントキーのファイル名
SERVICE_ACCOUNT_FILE = 'service_account.json'
# 3. 画像が保存されているGoogle DriveフォルダのID
DRIVE_FOLDER_ID = '1P0kb5GhoYKZKv0x8BOhFnOISLwFO-D2c'
# 4. Driveのスコープ（変更不要）
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
# ===========================


# モデルの読み込み
print("モデルを読み込んでいます...")
model = YOLO(MODEL_PATH)
print("モデル読み込み完了。")

# キャッシュ防止の設定
@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    return r

# --- Google Driveから最新画像をダウンロードする関数 ---
def download_latest_image_from_drive():
    try:
        # 認証を行う
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        service = build('drive', 'v3', credentials=creds)

        # 指定フォルダ内の画像ファイルを、更新日時が新しい順に検索
        # qパラメータで検索条件を指定（親フォルダID、MIMEタイプが画像、ゴミ箱に入っていない）
        results = service.files().list(
            q=f"'{DRIVE_FOLDER_ID}' in parents and mimeType contains 'image/' and trashed=false",
            orderBy="modifiedTime desc",
            pageSize=1, # 最新の1つだけ取得
            fields="files(id, name, modifiedTime)"
        ).execute()
        
        files = results.get('files', [])

        if not files:
            print("ドライブフォルダに画像が見つかりませんでした。")
            return None

        latest_file = files[0]
        print(f"最新の画像を見つけました: {latest_file['name']} (ID: {latest_file['id']})")

        # ファイルをダウンロードする準備
        request = service.files().get_media(fileId=latest_file['id'])
        fh = io.BytesIO() # メモリ上に一時保存
        downloader = MediaIoBaseDownload(fh, request)
        
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            # print(f"Download {int(status.progress() * 100)}%.")

        # ダウンロードしたデータをOpenCVで読み込める形式に変換
        fh.seek(0)
        file_bytes = np.asarray(bytearray(fh.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        return img

    except Exception as e:
        print(f"Google Drive接続エラー: {e}")
        return None

# --- AI解析用のバックグラウンド処理 ---
last_updated = "未更新"
import numpy as np # cv2.imdecodeで必要になるため追加

def update_analysis():
    global last_updated
    while True:
        print("\n--- 10分定期実行を開始します ---")
        # 1. Driveから最新画像をダウンロード
        frame = download_latest_image_from_drive()

        if frame is not None:
            print("画像の解析を開始します...")
            # 2. YOLOで推論
            results = model(frame)
            
            # staticフォルダの確認
            if not os.path.exists('static'):
                os.makedirs('static')
            
            # 3. 結果画像を保存
            annotated_frame = results[0].plot(line_width=2, font_size=1.0, conf=False)
       
            cv2.imwrite('static/analysis.jpg', annotated_frame)
            
            # 更新時刻を記録
            last_updated = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            print(f"AI解析完了し保存しました: {last_updated}")
        else:
            print("画像の取得に失敗したため、解析をスキップします。")

        print("次の実行まで10分待機します...")
        # 600秒（10分）待機
        time.sleep(600)

# スレッドを開始
# エラーでスレッドが落ちてもメインプロセスが続くようにtry-exceptで囲むとより安全ですが
# まずはこの形で動作確認を行います。
print("バックグラウンドスレッドを起動します...")
threading.Thread(target=update_analysis, daemon=True).start()

@app.route('/')
def index():
    return render_template('index.html', last_updated=last_updated)

if __name__ == '__main__':
    # debug=Trueだとスレッドが2重起動することがあるためFalse推奨
    app.run(host='0.0.0.0', port=5000, debug=False)