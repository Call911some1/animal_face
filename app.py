import streamlit as st
from PIL import Image
import os
import torch
import numpy as np
import requests
from ultralytics import YOLO
import cv2

# Функция загрузки модели для животных
def load_animal_model():
    model_path = "notebooks/yolov8_animals.pt"  # Путь к модели для животных
    model = YOLO(model_path)  # Загрузка модели YOLOv8
    return model

# Функция загрузки модели для лиц
def load_face_model():
    model_path = "notebooks/faces/facial_det.pt"  # Путь к модели для лиц
    model = YOLO(model_path)  # Загрузка модели YOLOv8
    return model

# Функция для выполнения детекции животных
def detect_animals(model, image):
    img = np.array(image)  # Преобразование изображения в numpy
    results = model(img)  # Вызов модели для детекции животных
    return results

# Функция для выполнения детекции лиц
def detect_faces(model, image):
    img_np = np.array(image)  # Преобразуем изображение в numpy массив
    results = model(img_np)  # Вызов модели для детекции лиц
    return results

# Функция для размытия лиц
def blur_faces(image, results):
    orig_img = np.array(image).copy()  # Копируем оригинальное изображение
    
    # Проходимся по bounding box каждого обнаруженного лица
    for box in results[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box)  # Преобразуем координаты в int

        # Вырезаем область лица
        roi = orig_img[y1:y2, x1:x2]
        
        # Применяем размытие к этой области
        blurred_roi = cv2.GaussianBlur(roi, (51, 51), 30)
        
        # Вставляем размытую область обратно в изображение
        orig_img[y1:y2, x1:x2] = blurred_roi

    return orig_img

# Основная функция Streamlit
def main():
    st.sidebar.title("Выбор модели")
    page = st.sidebar.selectbox("Выберите модель:", ['Animal Detection with YOLOv8', 'Обнаружение и размытие лиц'])

    if page == 'Animal Detection with YOLOv8':
        st.title("Animal Detection with YOLOv8")
        st.write("Это приложение использует модель YOLOv8 для детекции животных.")

        # Загрузка изображений с компьютера или по ссылке
        uploaded_files = st.file_uploader("Загрузите изображения", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
        link = st.text_input("Или введите ссылку на изображение")

        # Загрузка модели для животных
        model = load_animal_model()

        images = []

        # Если загружены файлы
        if uploaded_files:
            for file in uploaded_files:
                image = Image.open(file)
                images.append(image)
                st.image(image, caption=f"Загружено: {file.name}")

        # Если введена ссылка
        if link:
            try:
                img = Image.open(requests.get(link, stream=True).raw)
                images.append(img)
                st.image(img, caption="Загружено по ссылке")
            except:
                st.error("Ошибка загрузки изображения. Проверьте ссылку.")

        # Выполнение детекции объектов
        if images:
            for image in images:
                st.write("Результаты детекции:")
                results = detect_animals(model, image)

                # Рендерим результаты и показываем изображения с детекцией
                result_image = results[0].plot()  # Метод plot() визуализирует результаты детекции
                st.image(result_image, caption="Результаты детекции")  # Отображение детектированного изображения

            # Показ метрик модели
            st.subheader("Информация о модели и метриках")
            st.write("Модель: yolov8m.pt")
            st.write("Количество эпох: 100")
            st.write("Модель обучалась: 49 минут")
            st.write("Общий объем картинок: 1504")
            st.write("Объем картинок для train: 1048")
            st.write("Объем картинок для test: 228")
            st.write("Объем картинок для valid: 224")
            st.write("Метрики модели:")

            # Отображение PR-кривой и confusion matrix для животных
            st.image(os.path.join("notebooks", "runs", "detect", "train", "PR_curve.png"))
            st.image(os.path.join("notebooks", "runs", "detect", "train", "confusion_matrix.png"))

    elif page == 'Обнаружение и размытие лиц':
        st.title("Обнаружение и размытие лиц")
        st.write("Это приложение использует модель YOLOv8 для обнаружения и размытия лиц.")

        # Загрузка изображений с компьютера или по ссылке
        uploaded_files = st.file_uploader("Загрузите изображения", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
        link = st.text_input("Или введите ссылку на изображение")

        # Загрузка модели для лиц
        model = load_face_model()

        images = []

        # Если загружены файлы
        if uploaded_files:
            for file in uploaded_files:
                image = Image.open(file)
                images.append(image)
                st.image(image, caption=f"Загружено: {file.name}")

        # Если введена ссылка
        if link:
            try:
                img = Image.open(requests.get(link, stream=True).raw)
                images.append(img)
                st.image(img, caption="Загружено по ссылке")
            except:
                st.error("Ошибка загрузки изображения. Проверьте ссылку.")

        # Выполнение детекции и размытия лиц
        if images:
            for image in images:
                st.write("Результаты детекции:")
                results = detect_faces(model, image)  # Детекция лиц
                blurred_image = blur_faces(image, results)  # Размытие лиц
                
                # Отображение результата
                st.image(blurred_image, caption="Лица размыты")

        st.subheader("Информация о модели и метриках")
        st.write("Модель: facial_det.pt")
        st.write("Количество эпох: 50")
        st.write("Модель обучалась: 36 минут")
        st.write("Общий объем картинок: 16733")
        st.write("Объем картинок для train: 13386")
        st.write("Объем картинок для valid: 3347")

        # Здесь графики для лиц
        st.image(os.path.join("PR_curve.png"))
        st.image(os.path.join('confusion_matrix.png'))

# Запуск приложения
if __name__ == "__main__":
    main()
