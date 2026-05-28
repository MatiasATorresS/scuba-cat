import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import urllib.request
import time
import os
import math

# Descargar modelos si no existen
modelo_manos = 'hand_landmarker.task'
if not os.path.exists(modelo_manos):
    print("Descargando modelo de manos...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        modelo_manos
    )

modelo_cara = 'face_detector.tflite'
if not os.path.exists(modelo_cara):
    print("Descargando modelo de cara...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
        modelo_cara
    )

# Configurar detector de manos
opciones_manos = mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=modelo_manos),
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.75,
    min_hand_presence_confidence=0.75,
    min_tracking_confidence=0.75
)
detector_manos = mp_vision.HandLandmarker.create_from_options(opciones_manos)

# Configurar detector de cara
opciones_cara = mp_vision.FaceDetectorOptions(
    base_options=mp_python.BaseOptions(model_asset_path=modelo_cara),
    running_mode=mp_vision.RunningMode.VIDEO,
    min_detection_confidence=0.6
)
detector_cara = mp_vision.FaceDetector.create_from_options(opciones_cara)

def calc_dist(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def es_puño(lm):
    dedos = [(8, 5), (12, 9), (16, 13), (20, 17)]
    muñeca = lm[0]
    for punta, base in dedos:
        if calc_dist(lm[punta], muñeca) >= calc_dist(lm[base], muñeca) * 1.3:
            return False
    return True

def es_palma(lm):
    dedos = [(8, 5), (12, 9), (16, 13), (20, 17)]
    muñeca = lm[0]
    for punta, base in dedos:
        if calc_dist(lm[punta], muñeca) <= calc_dist(lm[base], muñeca):
            return False
    return True

def mano_cubre_cara(lm_mano, bbox_cara, margen=0.05):
    """
    Verifica si algún landmark de la mano está dentro del bounding box de la cara.
    margen: expande un poco el bbox para ser menos estricto
    """
    if bbox_cara is None:
        return False
    
    x1, y1, x2, y2 = bbox_cara
    
    # Revisar varios landmarks clave: muñeca(0), nudillos(5,9,13,17), palma(0)
    puntos_clave = [0, 5, 9, 13, 17, 8, 12, 16, 20]
    for i in puntos_clave:
        px = lm_mano[i].x
        py = lm_mano[i].y
        if (x1 - margen) < px < (x2 + margen) and (y1 - margen) < py < (y2 + margen):
            return True
    return False

conexiones = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

def dibujar_mano(img, lm):
    h, w, _ = img.shape
    puntos = [(int(pt.x * w), int(pt.y * h)) for pt in lm]
    for p1, p2 in conexiones:
        cv2.line(img, puntos[p1], puntos[p2], (0, 200, 255), 2)
    for p in puntos:
        cv2.circle(img, p, 4, (0, 100, 255), -1)

def dibujar_cara(img, bbox):
    if bbox is None:
        return
    h, w, _ = img.shape
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img,
        (int(x1 * w), int(y1 * h)),
        (int(x2 * w), int(y2 * h)),
        (0, 255, 0), 2
    )

cap = cv2.VideoCapture(0)
reproductor = cv2.VideoCapture('gato.mp4')

reproduciendo = False
tiempo_gracia = 0
inicio = time.time()
ultimo_bbox_cara = None  # Guardar el último bbox detectado

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ts = int((time.time() - inicio) * 1000)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    res_manos = detector_manos.detect_for_video(mp_img, ts)
    res_cara = detector_cara.detect_for_video(mp_img, ts)
    
    # Actualizar bbox de cara solo cuando se detecta
    # (si la mano la tapa, puede que no se detecte, usamos el último conocido)
    if res_cara.detections:
        det = res_cara.detections[0]
        bb = det.bounding_box
        h_f, w_f, _ = frame.shape
        ultimo_bbox_cara = (
            bb.origin_x / w_f,
            bb.origin_y / h_f,
            (bb.origin_x + bb.width) / w_f,
            (bb.origin_y + bb.height) / h_f
        )
    
    dibujar_cara(frame, ultimo_bbox_cara)
    
    gesto = "NINGUNO"
    
    if res_manos.hand_landmarks:
        landmarks = res_manos.hand_landmarks[0]
        dibujar_mano(frame, landmarks)
        
        if es_puño(landmarks):
            gesto = "PUNO"
        elif mano_cubre_cara(landmarks, ultimo_bbox_cara):
            gesto = "TAPANDO_CARA"
        elif es_palma(landmarks):
            gesto = "PALMA"
    
    cv2.putText(frame, f"Gesto: {gesto}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Reproducir si es puño O si está tapando la cara
    if gesto in ("PUNO", "TAPANDO_CARA"):
        tiempo_gracia = 0
        reproduciendo = True
    else:
        if reproduciendo:
            if tiempo_gracia == 0:
                tiempo_gracia = time.time()
            elif time.time() - tiempo_gracia > 0.5:
                reproduciendo = False
                tiempo_gracia = 0
                reproductor.set(cv2.CAP_PROP_POS_FRAMES, 0)
                cv2.destroyWindow("Reproductor Scuba")

    if reproduciendo and reproductor.isOpened():
        ok, frame_video = reproductor.read()
        if not ok:
            reproductor.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame_video = reproductor.read()
        if ok:
            frame_video = cv2.resize(frame_video, (450, 450))
            cv2.imshow("Reproductor Scuba", frame_video)

    cv2.imshow("Camara", frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
reproductor.release()
detector_manos.close()
detector_cara.close()
cv2.destroyAllWindows()