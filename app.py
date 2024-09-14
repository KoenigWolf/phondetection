from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import tensorflow as tf
import time
import os

# Flaskアプリケーションのインスタンスを作成
app = Flask(__name__)

# モデルとラベルマップのパス（相対パス）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'ssd_mobilenet_v2_coco_2018_03_29', 'frozen_inference_graph.pb')
LABELS_PATH = os.path.join(BASE_DIR, 'mscoco_label_map.pbtxt')

# ラベルマップの読み込み
with open(LABELS_PATH, 'r') as file:
    label_map = file.read()

category_index = {}
for line in label_map.split('\n'):
    if "id:" in line:
        category_id = int(line.split(': ')[1])
    if "display_name:" in line:
        category_name = line.split(': ')[1].strip().strip('\"')
        category_index[category_id] = category_name

# TensorFlowモデルの読み込み
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(MODEL_PATH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    # カメラキャプチャを開始
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "Error: Could not open video capture"

    photo_path = None
    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            while True:
                # フレームをキャプチャ
                ret, frame = cap.read()
                if not ret:
                    return "Error: Failed to grab frame"

                # フレームのサイズを取得
                height, width = frame.shape[:2]

                # 入力テンソルの準備
                image_np_expanded = np.expand_dims(frame, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                # 検出を実行
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # 検出結果を解析
                for i in range(int(num_detections[0])):
                    if scores[0][i] > 0.5:
                        class_id = int(classes[0][i])
                        if category_index[class_id] == "cell phone":
                            # バウンディングボックスを取得
                            box = boxes[0][i] * np.array([height, width, height, width])
                            (startY, startX, endY, endX) = box.astype("int")

                            # バウンディングボックスを描画
                            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

                            # 写真を保存
                            timestamp = time.strftime("%Y%m%d-%H%M%S")
                            photo_filename = f'photo_{timestamp}.png'
                            photo_path = os.path.join('static', photo_filename)
                            print(f"Saving photo to: {photo_path}")  # デバッグ用

                            # ファイルを保存
                            if cv2.imwrite(photo_path, frame):
                                print(f"Photo saved successfully: {photo_path}")
                            else:
                                print(f"Failed to save photo: {photo_path}")

                            cap.release()
                            cv2.destroyAllWindows()
                            return redirect(url_for('show_photo', photo_filename=photo_filename))

                # 'q'キーで終了（手動で）
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # リリースとウィンドウの破棄
            cap.release()
            cv2.destroyAllWindows()

    return "No phone detected."

@app.route('/show_photo')
def show_photo():
    photo_filename = request.args.get('photo_filename')
    photo_path = url_for('static', filename=photo_filename)
    print(f"Photo path: {photo_path}")  # デバッグ用
    return render_template('show_photo.html', photo_path=photo_path)

if __name__ == "__main__":
    app.run(debug=True)