 # Онлайн-деканат

<!-- Статус проекта -->
[![Status](https://img.shields.io/badge/Status-в_разработке-orange?style=for-the-badge&logo=git)](https://shields.io/)

[![C++](https://img.shields.io/badge/C%2B%2B-17%2F20-00599C?style=for-the-badge&logo=cplusplus&logoColor=white)](https://isocpp.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-C%2B%2B_API-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Tesseract OCR](https://img.shields.io/badge/Tesseract-OCR_C%2B%2B-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://github.com/tesseract-ocr/tesseract)
[![CMake](https://img.shields.io/badge/CMake-Build-064F8C?style=for-the-badge&logo=cmake&logoColor=white)](https://cmake.org/)

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![pybind11](https://img.shields.io/badge/pybind11-Binding-FF6F00?style=for-the-badge&logo=python&logoColor=white)](https://github.com/pybind/pybind11)

[![MS SQL Server](https://img.shields.io/badge/MS_SQL_Server-T--SQL-CC292B?style=for-the-badge&logo=microsoftsqlserver&logoColor=white)](https://www.microsoft.com/sql-server)
[![Platform](https://img.shields.io/badge/Platform-macOS_%7C_Windows-lightgrey?style=for-the-badge&logo=apple&logoColor=black)](https://shields.io/)

«Онлайн-деканат» — это интеллектуальный цифровой помощник в виде кроссплатформенного десктопного приложения (macOS и Windows), который сканирует студенческий документ с помощью **веб-камеры устройства**, распознает текст через OCR и мгновенно отвечает на вопросы, основываясь на актуальных правилах и регламентах вуза.

## Технологии и стек
Полный технологический стек проекта:

- **C++ ( C++)** — высокопроизводительное ядро приложения для обработки видеопотока с веб-камеры, детекции границ документов и предобработки изображений
- **OpenCV (C++ API)** — библиотека компьютерного зрения для захвата кадров, выравнивания перспективы, HSV-фильтрации синих печатей и подготовки ROI к OCR
- **Tesseract OCR (C++ API) / EasyOCR** — движок распознавания печатного и рукописного текста на студенческих билетах
- **Python 3.10+** — язык для реализации бизнес-логики, связки компонентов и backend-сервисов
- **FastAPI** — веб-фреймворк для создания REST API бэкенда
- **CMake & pybind11** — сборка C++ модулей и их интеграция/биндинг в Python-окружение
- **MS SQL Server (T-SQL)** — реляционная база данных для хранения данных студентов, правил, регламентов и истории запросов
- **pyodbc / SQLAlchemy** — инструменты для подключения и работы с MS SQL Server
- **CustomTkinter / PyQt6** — графический десктопный интерфейс приложения (macOS / Windows)
- **Git, CMake, VS Code / JetBrains CLion / Rider** — инструменты разработки и сборки


## Содержание
- [Технологии и стек](#технологии-и-стек)

- [Структура проекта и задачи](#структура-проекта-и-задачи)

- [Использование](#использование)

- [Начало работы](#начало-работы)

- [Тестирование](#тестирование)

- [Deploy и CI/CD](#deploy-и-cicd)

- [Contributing](#contributing)

- [FAQ](#faq)

- [To do](#to-do)

- [Команда проекта](#команда-проекта)
