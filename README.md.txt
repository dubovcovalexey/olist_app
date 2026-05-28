# Интеллектуальная рекомендательная система Olist со скорингом

Рекомендательная система построена на бразильском датасете Olist Ecommerce. Включает 5 моделей: Alibaba Swing, Content-Based, Score Fusion, Two-Tower (PyTorch) и CatBoostRanker.

## Запуск проекта через Docker

Чтобы запустить приложение в изолированном контейнере, выполните в терминале папки проекта следующие команды:

1. **Сборка Docker-образа:**
```bash
docker build -t olist-rec-app .
```

2. **Запуск Docker-контейнера:**
```bash
docker run -p 8501:8501 olist-rec-app
```

После старта контейнера откройте браузер по адресу: `http://localhost:8501`
