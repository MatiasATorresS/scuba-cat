# Scuba Cat 🐱🤿
 
Reproduce un video de un gato buceador cuando tapas tu cara o haces un puño frente a la cámara.
 
## Setup
 
```bash
pip install opencv-python mediapipe
```
 
Pon tu `gato.mp4` en la misma carpeta y listo. Los modelos de MediaPipe se descargan solos la primera vez.
 
## Uso
 
```bash
python scuba-cat.py
```
 
`ESC` para salir.
 
## Problemas comunes
 
- Si no agarra bien los gestos, mejora la iluminación
- Si no usa la cámara correcta, cambia el `0` en `cv2.VideoCapture(0)` por `1` o `2`