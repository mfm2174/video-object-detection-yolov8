# 1ero CÓDIGO PARA LER CONTEÚDO DE VÍDEO E GERA OUTRO VÍDEO INDICANDO O TIPO DE OBJETO
# Modelo Simples

!pip install ultralytics opencv-python

from google.colab import drive
from ultralytics import YOLO
import cv2

drive.mount('/content/drive', force_remount=True)

video_path="/content/drive/MyDrive/Colab Notebooks/.../Video1.mp4"

# === Carregar modelo YOLO ===
model = YOLO("yolov8n.pt")

# === Abrir vídeo ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_path = "/content/drive/MyDrive/Colab Notebooks/.../resultado_yolo1.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

class_counts = {}

print("Processando vídeo... aguarde!")

# === Loop pelos frames ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    boxes = results[0].boxes

    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])

        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, max(y1-10,0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    out.write(frame)

cap.release()
out.release()

print("\n✔ Vídeo processado!")
print("Arquivo criado em:", output_path)

print("\n📊 Objetos detectados:")
for name, count in class_counts.items():
    print(f"- {name}: {count}")
