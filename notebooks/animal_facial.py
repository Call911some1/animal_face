import streamlit as st
from PIL import Image
import os
import torch
import numpy as np
import requests
from ultralytics import YOLO
import cv2
import gdown

page = st.sidebar.selectbox(":blue[Выберите страницу]", ['Animal Detection with YOLOv8', 'Обнаружение человеческих лиц'])

# Проверка на текущую страницу
if page == 'Animal Detection with YOLOv8':
    # Функция загрузки модели
    def load_model():
        model_path = 'notebooks/facial_det.pt'  # Указываем путь к модели
        model = YOLO(model_path)  # Загрузка модели YOLOv8
        return model

    # Функция для выполнения детекции объектов (животных)
    def detect_objects(model, image):
        img = np.array(image)  # Преобразование изображения в numpy
        results = model(img)  # Вызов модели для детекции
        return results

    # Основная страница
    def main():
        st.title("Animal Detection with YOLOv8")
        st.write("Это приложение использует модель YOLOv8 для детекции животных.")

        # Загрузка изображений с компьютера или по ссылке
        uploaded_files = st.file_uploader("Загрузите изображения", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
        link = st.text_input("Или введите ссылку на изображение")

        # Загрузка модели для животных
        model = load_model()

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
                results = detect_objects(model, image)

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

            # Отображение PR-кривой и confusion matrix
            st.image(os.path.join("notebooks", "runs", "detect", "train", "PR_curve.png"))
            st.image(os.path.join("notebooks", "runs", "detect", "train", "confusion_matrix.png"))

    # Запуск приложения
    if __name__ == "__main__":
        main()
if page == 'Обнаружение человеческих лиц':
    def load_model():
        # model_path = "/home/hopelessdreamer/ds_bootcamp/projects/fac_det_project/face_or_animal/notebooks/facial_det.pt"
        model_path = "notebooks/yolov8_animals.pt"  
        model = YOLO(model_path)  
        return model

    def detect_objects(model, image):
        img = np.array(image)  # Преобразование изображения в numpy
        results = model(img)  # Вызов модели для детекции
        return results

    # Основная страница
    def main():
        st.title(':grey[Обнаружение человеческих лиц]')
        st.write('***:violet[Есть ли здесь люди на фото? Если есть — замажем!]***')

        # Загрузка изображений с компьютера или по ссылке
        uploaded_files = st.file_uploader("Загрузите изображения", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
        link = st.text_input("Или введите ссылку на изображение")

        # Загрузка модели для животных
        model = load_model()

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
                img_np = np.array(image)  # Преобразуем изображение в numpy массив
                results = detect_objects(model, image)  # Получаем результаты детекции

                # Получаем оригинальное изображение из результатов
                orig_img = results[0].orig_img.copy()  # Делаем копию оригинального изображения для работы

                # Получаем bbox и конвертируем в numpy массив
                boxes = results[0].boxes.xyxy.cpu().numpy()  # Преобразуем bbox в numpy массив
                
                # Размываем только области внутри bbox
                for box in boxes:
                    x1, y1, x2, y2 = box  # Распаковываем координаты bbox и другие параметры
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Преобразуем в int
                    
                    # Вырезаем область изображения с bbox
                    roi = orig_img[y1:y2, x1:x2]  # Получаем область интереса (ROI)

                    # Применяем GaussianBlur к этой области
                    blurred_roi = cv2.GaussianBlur(roi, (91, 91), 20)

                    # Вставляем размытую область обратно в оригинальное изображение
                    orig_img[y1:y2, x1:x2] = blurred_roi  # Заменяем оригинальную область на размытую

                # Рендерим результаты и показываем изображения с детекцией
                st.image(orig_img, caption="Результаты детекции") 

        st.subheader("Информация о модели и метриках")
        st.write("Модель: yolov5n")
        st.write("Количество эпох: 3")
        st.write("Модель обучалась: 36 минут")
        st.write("Общий объем картинок: 16733")
        st.write("Объем картинок для train: 13386")
        st.write("Объем картинок для valid: 3347")
        st.write("Метрики модели:")

        # # Отображение PR-кривой и confusion matrix
        # st.image(os.path.join("notebooks", "runs", "detect", "train", "PR_curve.png"))
        # st.image(os.path.join("notebooks", "runs", "detect", "train", "confusion_matrix.png"))
        
    # Запуск приложения
    if __name__ == "__main__":
        main()