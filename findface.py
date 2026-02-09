import cv2
import torch
import torch.nn.functional as f
import torchvision.transforms as transforms
from backbone import Backbone
from PIL import Image
import os
import glob
from config import (
    camera_index,
    frame_skip,
    detection_frame_size,
    model_det_path
)

class FaceDetector:
    def __init__(self):
        # Инициализация камеры
        self.cap  = cv2.VideoCapture(camera_index)

        # Параметры оптимизации
        self.frame_skip = frame_skip
        self.frame_count = 0
        self.detection_frame_size = detection_frame_size

        # Загрузка ONNX модели для детекции лиц
        self.detector = cv2.FaceDetectorYN.create(
            model=model_det_path,
            config="",
            input_size=detection_frame_size,
            score_threshold=0.5,
            nms_threshold=0.3,
            top_k=10
        )

    def run(self):
        # Основной цикл обработки видеопотока
        while True:
            # Захват кадра
            ret, frame = self.cap.read()
            
            if not ret:
                print("Не удалось получить кадр с камеры")
                break
            
            # Пропуск кадров в соответствии
            self.frame_count += 1
            if self.frame_count % self.frame_skip == 0:
                # Детекция лиц только в каждом н-ом кадре
                faces = self.detect_faces(frame)

                # Проверяем, есть ли лица на кадре, и отправляем уведомление
                if faces:
                    self.send_notification(f"Обнаружено {len(faces)} лиц(а) на кадре")
                    print(f"Обнаружено {len(faces)} лиц(а) на кадре")
                else:
                    print("Лица не обнаружены")

                # Сбрасываем счетчик, чтобы не вызывать переполнение
                self.frame_count = 0
            
            # Отображение результата
            cv2.imshow('Face Detection', frame)
            
            # Обработка нажатия клавиш
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Клавиши 'q' или 'Esc' для выхода
                break

        
        # Освобождение ресурсов
        self.cap.release()


    def detect_faces(self, frame):
        # Детекция лиц на кадре

        # Подготовка изображения для детектора YOLO
        # Изменение размера изображения до нужного размера
        resized_frame = cv2.resize(frame, self.detection_frame_size)
        
        # Детекция лиц
        faces = self.detector.detect(resized_frame)
        
        # Преобразование результатов в формат, совместимый с dlib
        detected_faces = []
        if faces[1] is not None:
            for detection in faces[1]:
                # Получаем координаты и размеры прямоугольника
                x, y, w, h = int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3])
                # Преобразуем координаты в исходный размер кадра
                x = int(x * frame.shape[1] / self.detection_frame_size[0])
                y = int(y * frame.shape[0] / self.detection_frame_size[1])
                w = int(w * frame.shape[1] / self.detection_frame_size[0])
                h = int(h * frame.shape[0] / self.detection_frame_size[1])
                
                # Создаем объект, имитирующий dlib.rectangle
                face_rect = type('FaceRect', (), {
                    'left': lambda self, x=x: x,
                    'top': lambda self, y=y: y,
                    'right': lambda self, x=x, w=w: x + w,
                    'bottom': lambda self, y=y, h=h: y + h,
                    'width': lambda self, w=w: w,
                    'height': lambda self, h=h: h
                })()
                
                detected_faces.append(face_rect)
        
        return detected_faces

    def send_notification(self, message):
        # Отправка уведомления при обнаружении лица
        print(message)