import requests
import json

url = "http://localhost:8000/ask"

payload = {
    "question": "есть фильм интересный фантастика на вечер?",
    "k": 3
}


print("Отправка тестового запроса к ИИ-Агенту через локальный хост...")

try:
    session = requests.Session()
    session.trust_env = False  # Игнорируем прокси Windows
    
    # Увеличиваем таймаут до 60 секунд, чтобы CPU успел всё обсчитать
    response = session.post(url, json=payload, timeout=60)
    
    # Читаем как сырой текст, requests сам разберется с кодировкой ответа
    data = response.json()
    
    print("\n🎉 Ответ от ИИ-Агента:")
    print(json.dumps(data, ensure_ascii=False, indent=2))
    
except Exception as e:
    print(f"\n❌ Ошибка при отправке запроса: {e}")
