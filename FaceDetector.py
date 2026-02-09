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
    frame_width,
    frame_height,
    frame_skip,
    detection_frame_size,
    model_det_path,
    model_rec_path,
    data_root,
    cos_sim_threshold
)

from camera_notifier import notify_camera_blocked

class FaceDetector:
    def __init__(self):
        # Инициализация камеры
        self.cap  = cv2.VideoCapture(camera_index)
        
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

        # Параметры оптимизации
        self.frame_skip = frame_skip
        self.frame_count = 0
        self.detection_frame_size = detection_frame_size
        
        # Переменная для хранения последних обнаруженных лиц
        self.last_detected_faces_with_similarities = []
        
        # Счетчик для сохраненных файлов
        self.saved_faces_count = self.get_next_face_count()
        
        # Переменные для отслеживания нераспознанных лиц
        self.previous_unknown_faces_count = 0
        self.unknown_faces_current_count = 0
        self.unknown_faces_appear_time = None  # Время появления нераспознанных лиц

        # Путь к папке для сохранения нераспознанных лиц
        from config import foreign_faces_path
        self.foreignfaces_path = foreign_faces_path

        # Для отслеживания времени последней проверки новых файлов
        import time
        self.last_check_time = time.time()
        self.check_interval = 10  # интервал проверки новых файлов в секундах



        
        # Переменная для отслеживания состояния камеры
        self.camera_blocked = False
        
        # Загрузка ONNX модели для детекции лиц
        self.detector = cv2.FaceDetectorYN.create(
            model=model_det_path,
            config="",
            input_size=detection_frame_size,
            score_threshold=0.5,
            nms_threshold=0.3,
            top_k=10
        )

        # Загрузка модели для получения эмбеддингов (кэширование модели)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.backbone = Backbone([112, 112])
        self.backbone.load_state_dict(torch.load(model_rec_path, map_location=torch.device("cpu")))
        self.backbone.to(self.device)
        self.backbone.eval()
        
        # Загружаем эталонные эмбеддинги из файлов в директории data_root
        self.reference_embeddings = []
        reference_files = glob.glob(os.path.join(data_root, "*.jpg"))
        
        processed_files = set()
        for ref_file in reference_files:
            try:
                # Извлекаем имя файла без пути
                filename = os.path.basename(ref_file)
                
                # Проверяем, не обрабатывали ли мы этот файл ранее
                if filename not in processed_files:
                    pil_face = self.get_ref_face(ref_file)
                    embedding = self.get_embedding(pil_face, backbone=self.backbone, device=self.device)
                    self.reference_embeddings.append(embedding)
                    processed_files.add(filename)
            except ValueError as e:
                print(f"Ошибка при обработке файла {ref_file}: {e}")
                continue

        # Преобразуем список в тензор
        if self.reference_embeddings:
            self.reference_embeddings = torch.stack(self.reference_embeddings)
            print(f"Загружено эталонных эмбеддингов при инициализации: {len(self.reference_embeddings)}")
        else:
            print("Не найдено эталонных изображений в директории data_root")
            self.reference_embeddings = torch.empty(0, 512)  # Пустой тензор, если нет эталонов

        # Создаем папку data_root, если она не существует
        if not os.path.exists(data_root):
            os.makedirs(data_root)

    def is_camera_blocked(self, frame):
        # Проверка, загорожена ли камера
        # Определяем по однородности фона
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Вычисляем стандартное отклонение яркости (однородность)
        std_dev = cv2.meanStdDev(gray)[1][0][0]
        
        # Если стандартное отклонение слишком низкое, это означает однородный фон
        # что может говорить о загораживающем объекте
        if std_dev < 25:  # Порог однородности
            return True
        
        # Также проверим, если изображение слишком темное
        mean_brightness = cv2.mean(gray)[0]
        if mean_brightness < 20:
            return True
        
        return False
    
    def run(self):
        # Основной цикл обработки видеопотока
        while True:
            # Захват кадра
            ret, frame = self.cap.read()
            
            if not ret:
                print("Не удалось получить кадр с камеры")
                break
            
            # Проверяем, загорожена ли камера
            is_blocked = self.is_camera_blocked(frame)
            if is_blocked and not self.camera_blocked:
                print("УВЕДОМЛЕНИЕ: Камера загорожена!")
                notify_camera_blocked()
                self.camera_blocked = True
            elif not is_blocked and self.camera_blocked:
                # Камера снова доступна - сбрасываем флаг, но не выводим уведомление
                self.camera_blocked = False
            
            # Пропуск кадров в соответствии с параметром frame_skip
            self.frame_count += 1
            if self.frame_count % self.frame_skip == 0:
                # Детекция лиц только в каждом N-ом кадре (где N = frame_skip)
                faces = self.detect_faces(frame)
                face_similarities = self.calculate_face_similarities(frame, faces)
                self.draw_faces(frame, face_similarities)

                # Сохраняем последние обнаруженные лица и их сходства в виде списка словарей
                self.last_detected_faces_with_similarities = []
                for item in face_similarities:
                    self.last_detected_faces_with_similarities.append(item)

                # Проверяем и сохраняем скриншот с нераспознанными лицами (с уже нарисованными прямоугольниками)
                unknown_faces_count = sum(1 for item in face_similarities if item['similarity'] < cos_sim_threshold)
                
                import time
                
                # Если количество нераспознанных лиц изменилось
                if unknown_faces_count != self.previous_unknown_faces_count:
                    # Если количество увеличилось, фиксируем время
                    if unknown_faces_count > self.previous_unknown_faces_count:
                        self.unknown_faces_appear_time = time.time()
                    self.previous_unknown_faces_count = unknown_faces_count
                elif unknown_faces_count > 0 and self.unknown_faces_appear_time is not None:
                    # Если количество не изменилось, но есть нераспознанные лица и был зафиксирован момент их появления
                    elapsed_time = time.time() - self.unknown_faces_appear_time
                    # Сохраняем скриншот, если прошло 0.5 секунды с момента появления нераспознанных лиц
                    if elapsed_time >= 0.5:
                        self.save_unknown_faces_screenshot(frame)
                        # Сбрасываем время, чтобы не сохранять повторно
                        self.unknown_faces_appear_time = None

                # Сбрасываем счетчик, чтобы не вызывать переполнение
                self.frame_count = 0
            else:
                # Отображение кадра с последними обнаруженными лицами для плавности видео
                self.draw_last_faces(frame)
            
            # Проверяем наличие новых файлов в папке data и обновляем эмбеддинги
            self.check_and_update_reference_embeddings()
            
            # Отображение результата
            cv2.imshow('Face Detection', frame)
            
            # Обработка нажатия клавиш
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and faces:  # Нажата клавиша 's' и есть обнаруженные лица
                # Сохраняем все обнаруженные лица
                for face in faces:
                    self.save_face(frame, face)
            elif key == ord('q') or key == 27:  # Клавиши 'q' или 'Esc' для выхода
                break
            # Проверка закрытия окна
            if cv2.getWindowProperty('Face Detection', cv2.WND_PROP_VISIBLE) < 1:
                break
        
        # Освобождение ресурсов
        self.cap.release()
        cv2.destroyAllWindows()
    
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
    
    def calculate_face_similarities(self, frame, faces):
        # Проверяем, загорожена ли камера
        if self.camera_blocked:
            # Если камера загорожена, не выполняем распознавание и не выводим уведомления о лицах
            return []
        
        # Вычисление сходства для каждого лица
        result = []
        recognized_count = 0
        unrecognized_count = 0
        
        for face in faces:
            cosine_sim = self.recognize_face_with_similarity(frame, face)
            result.append({'face': face, 'similarity': cosine_sim})
            
            # Подсчет распознанных и нераспознанных лиц
            if cosine_sim >= cos_sim_threshold:
                recognized_count += 1
            else:
                unrecognized_count += 1
        
        # Вывод информации о распознанных и нераспознанных лицах
        print(f"Распознанные лица: {recognized_count}, Нераспознанные лица: {unrecognized_count}")
        
        return result
    
    def draw_faces(self, frame, face_similarities):
        # Отрисовка прямоугольников вокруг лиц
        for item in face_similarities:
            face = item['face']
            cosine_sim = item['similarity']
            
            # Получение координат прямоугольника
            x = face.left()
            y = face.top()
            w = face.width()
            h = face.height()
            
            # Определение результата сходства
            is_match = cosine_sim >= cos_sim_threshold
            
            # Выбор цвета в зависимости от результата сравнения
            color = (0, 255, 0) if is_match else (0, 0, 255)  # Зеленый если совпадение, красный если нет
            
            # Отрисовка прямоугольника
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Отображение значения similarity рядом с прямоугольником
            similarity_text = f"Sim: {cosine_sim:.2f}"
            cv2.putText(frame, similarity_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Отображение инструкции для сохранения лица
        cv2.putText(frame, "Press 's' to save face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def draw_last_faces(self, frame):
        # Отрисовка прямоугольников вокруг последних обнаруженных лиц
        self.draw_faces(frame, self.last_detected_faces_with_similarities)

    def recognize_face_with_similarity(self, frame, face):
        # Сравнение распознанного лица со списком доверенных
        
        # Получение координат прямоугольника
        x = face.left()
        y = face.top()
        x2 = x + face.width()
        y2 = y + face.height()

        # Вырезаем область лица из кадра
        face_image = frame[y:y2, x:x2]

        # Проверяем, что изображение лица не пустое
        if face_image.size == 0:
            return 0.0

        # Преобразуем изображение в формат PIL
        face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_image_rgb)

        # Получаем эмбеддинг, используя кэшированную модель
        embedding = self.get_embedding(face_pil, 512, [112, 112], backbone=self.backbone, device=self.device)

        # Проверяем, есть ли эталонные эмбеддинги
        if self.reference_embeddings.numel() == 0:  # Если тензор пустой
            return 0.0  # Возвращаем 0.0, что означает отсутствие совпадения

        # Вычисляем косинусное сходство с каждым эталонным эмбеддингом
        # Умножаем на транспонированный тензор эталонных эмбеддингов для получения сходства со всеми эталонами
        cosine_sims = torch.matmul(self.reference_embeddings, embedding.unsqueeze(1)).squeeze(1).clip(min=0, max=1)

        # Возвращаем максимальное значение сходства
        max_cosine_sim = torch.max(cosine_sims).item()

        return max_cosine_sim
        
    def get_embedding(self, pil_image, embedding_size=512, input_size=[112, 112], backbone=None, device=None):
        """Получение эмбеддинга из изображения лица"""

        # Определяем процесс обработки изображения
        transform = transforms.Compose(
            [
                transforms.Resize(
                    [int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)],
                ),  # smaller side resized
                transforms.CenterCrop([input_size[0], input_size[1]]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ],
        )

        # Применение трансформаций
        image_tensor = transform(pil_image)  # Размер: [3, 112, 112]

        # Добавление размерности batch (необходимо для модели)
        image_tensor = image_tensor.unsqueeze(0)  # Размер: [1, 3, 112, 112]

        # Если backbone и device не переданы, создаем их (для обратной совместимости)
        if backbone is None or device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            backbone = Backbone(input_size)
            backbone.load_state_dict(torch.load(model_rec_path, map_location=torch.device("cpu")))
            backbone.to(device)
            backbone.eval()

        # Перемещение на устройство и получение эмбеддинга
        image_tensor = image_tensor.to(device)

        # Получение эмбеддинга (предполагается, что backbone уже загружен и переведен в eval режим)
        with torch.no_grad():
            embedding = torch.nn.functional.normalize(backbone(image_tensor)).cpu()

        # Удаляем размерности размером 1
        embedding = torch.squeeze(embedding)

        # Нормализуем векторы (если еще не нормализованы)
        embedding = embedding / torch.norm(embedding)

        return embedding

    def get_ref_face(self, file_name):
        # Загрузка и обработка эталонного изображения лица

        # Загрузка изображения для сравнения (опционально)
        image_pil = Image.open(file_name).convert('RGB')
        image_cv = cv2.imread(file_name)

        # Подготовка изображения для детектора YOLO
        resized_frame = cv2.resize(image_cv, self.detection_frame_size)

        # Детекция лица на изображении
        reference_faces = self.detector.detect(resized_frame)
        if reference_faces[1] is not None and len(reference_faces[1]) > 0:
            # Извлечение первого обнаруженного лица
            face_data = reference_faces[1][0]
            x, y, w, h = int(face_data[0]), int(face_data[1]), int(face_data[2]), int(face_data[3])
            # Преобразуем координаты в исходный размер изображения
            x = int(x * image_cv.shape[1] / self.detection_frame_size[0])
            y = int(y * image_cv.shape[0] / self.detection_frame_size[1])
            w = int(w * image_cv.shape[1] / self.detection_frame_size[0])
            h = int(h * image_cv.shape[0] / self.detection_frame_size[1])
            
            pil_face = image_pil.crop((x, y, x + w, y + h))

            return pil_face

        else:
            raise ValueError("Не удалось обнаружить лицо на эталонном изображении")
    
    def save_face(self, frame, face):
        # Сохранение лица в папку data в формате jpg
        if not os.path.exists(data_root):
            os.makedirs(data_root)
        
        # Получение координат прямоугольника
        x = face.left()
        y = face.top()
        w = face.width()
        h = face.height()
        
        # Вырезаем область лица из кадра
        face_image = frame[y:y+h, x:x+w]
        
        # Проверяем, что изображение лица не пустое
        if face_image.size == 0:
            print("Не удалось вырезать изображение лица")
            return False
        
        # Формирование имени файла
        self.saved_faces_count += 1
        filename = os.path.join(data_root, f"saved_face_{self.saved_faces_count}.jpg")
        
        # Сохранение изображения
        success = cv2.imwrite(filename, face_image)
        if success:
            print(f"Изображение лица сохранено: {filename}")
        else:
            print(f"Ошибка при сохранении изображения лица: {filename}")
        
        return success

    def save_unknown_faces_screenshot(self, frame):
        """Сохраняет скриншот с нераспознанными лицами в папку foreign"""
        import datetime
        
        # Формируем имя файла с временной меткой
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.foreignfaces_path, f"unknown_faces_{timestamp}.jpg")
        
        # Сохраняем кадр
        success = cv2.imwrite(filename, frame)
        if success:
            print(f"Скриншот с нераспознанными лицами сохранен: {filename}")
            return filename
        else:
            print(f"Ошибка при сохранении скриншота: {filename}")
            return None

    def get_next_face_count(self):
        # Получаем номер последнего сохраненного лица
        if not os.path.exists(data_root):
            return 0
        
        # Получаем все файлы с маской saved_face_*.jpg
        saved_faces = glob.glob(os.path.join(data_root, "saved_face_*.jpg"))
        
        # Извлекаем номера из имен файлов и находим максимальный
        max_count = 0
        for face_file in saved_faces:
            filename = os.path.basename(face_file)
            try:
                # Извлекаем число из имени файла
                count = int(filename.replace("saved_face_", "").replace(".jpg", ""))
                max_count = max(max_count, count)
            except ValueError:
                # Пропускаем файлы с неправильным форматом имени
                continue
        return max_count

    def check_and_update_reference_embeddings(self):
        """Проверяет наличие новых файлов в папке data и обновляет эталонные эмбеддинги"""
        import time
        
        current_time = time.time()
        
        # Проверяем, прошло ли достаточно времени с последней проверки
        if current_time - self.last_check_time < self.check_interval:
            return False
            
        self.last_check_time = current_time
        
        # Получаем список всех jpg файлов в папке data
        reference_files = glob.glob(os.path.join(data_root, "*.jpg"))
        
        # Сравниваем имена файлов с уже загруженными эмбеддингами
        current_filenames = set(os.path.basename(f) for f in reference_files)
        
        # Получаем количество загруженных эмбеддингов
        loaded_embedding_count = self.reference_embeddings.size(0) if hasattr(self.reference_embeddings, 'size') else len(self.reference_embeddings)
        
        # Если количество файлов изменилось, обновляем все эмбеддинги
        if len(current_filenames) != loaded_embedding_count:
            print(f"Обнаружено изменение в папке data: {len(current_filenames)} файлов (было {loaded_embedding_count})")
            
            # Загружаем все эталонные эмбеддинги заново
            all_embeddings = []
            processed_files = set()
            
            for ref_file in reference_files:
                try:
                    # Извлекаем имя файла без пути
                    filename = os.path.basename(ref_file)
                    
                    # Проверяем, не обрабатывали ли мы этот файл ранее
                    if filename not in processed_files:
                        pil_face = self.get_ref_face(ref_file)
                        embedding = self.get_embedding(pil_face, backbone=self.backbone, device=self.device)
                        all_embeddings.append(embedding)
                        processed_files.add(filename)
                        
                except ValueError as e:
                    print(f"Ошибка при обработке файла {ref_file}: {e}")
                    continue
            
            # Обновляем эталонные эмбеддинги
            if all_embeddings:
                self.reference_embeddings = torch.stack(all_embeddings)
                print(f"Обновлено эталонных эмбеддингов: {len(all_embeddings)}")
                return True
            else:
                print("Не удалось обработать ни один файл")
                
        return False
