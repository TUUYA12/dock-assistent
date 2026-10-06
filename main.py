import os
import re
import shutil
import joblib
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

# =====================================================================
# 1. КОНСТАНТЫ И НАСТРОЙКА КЛЮЧЕЙ
# =====================================================================
load_dotenv()  # Автоматически подгружает API_KEY из файла .env

CSV_PATH = "films_data.csv"                  # Локальный датасет (143 МБ)
INPUT_FOLDER = "new_documents"                 # Папка для поступающих документов
MODEL_PATH = "genre_classifier_model.pkl"       
VECTORIZER_PATH = "hashing_vectorizer.pkl"       

# Инициализация клиента OpenAI / GitHub Models
# Если используете GitHub Models Marketplace, оставьте base_url как указано ниже
client = OpenAI(
    base_url="https://azure.com", 
    api_key=os.getenv("API_KEY")
)

# =====================================================================
# 2. ПАЙПЛАЙН ОЧИСТКИ, УКРУПНЕНИЯ И ОБУЧЕНИЯ ЛОКАЛЬНОГО ИИ
# =====================================================================
def collapse_genres(raw_genre):
    if not isinstance(raw_genre, str): return None
    raw_genre = raw_genre.lower()
    if any(w in raw_genre for w in ['фантаст', 'фэнтези', 'космос', 'киберпанк']): return 'фантастика и фэнтези'
    elif any(w in raw_genre for w in ['ужас', 'хоррор', 'триллер', 'саспенс']): return 'ужасы и триллеры'
    elif any(w in raw_genre for w in ['комеди', 'сатир', 'юмор']): return 'комедия'
    elif any(w in raw_genre for w in ['боевик', 'экшен', 'детектив', 'кримин', 'преступ']): return 'боевик и детектив'
    elif any(w in raw_genre for w in ['истори', 'военн', 'война', 'биограф']): return 'история и война'
    elif any(w in raw_genre for w in ['документаль', 'хроника', 'научн']): return 'документальный'
    elif any(w in raw_genre for w in ['драма', 'мелодрама', 'романти']): return 'драма и мелодрама'
    return None

def clean_and_normalize_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^а-яёa-z0-9\s]', ' ', text)
    words = text.split()
    cleaned_tokens = [w for w in words if len(w) > 2]
    return " ".join(cleaned_tokens)

def train_local_classifier():
    """Мгновенно обучает линейную модель на CPU за 20 секунд."""
    print("⏳ Загрузка и очистка датасета (143 МБ)...")
    df = pd.read_csv(CSV_PATH)
    df['clean_genre'] = df['genre'].apply(collapse_genres)
    df = df.dropna(subset=['plot', 'clean_genre'])
    df['clean_plot'] = df['plot'].apply(clean_and_normalize_text)
    df = df[df['clean_plot'].str.len() >= 50].drop_duplicates(subset=['clean_plot'])
    
    print(f"✅ Выборка подготовлена: {len(df)} строк. Начинаем хэш-векторизацию...")
    vectorizer = HashingVectorizer(ngram_range=(1, 2), n_features=2**19, alternate_sign=False)
    X_hash = vectorizer.transform(df['clean_plot'])
    
    model = SGDClassifier(loss='log_loss', class_weight='balanced', max_iter=1000, random_state=42, n_jobs=-1)
    model.fit(X_hash, df['clean_genre'])
    
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print("💾 Локальная модель успешно зафиксирована на диске!")

# =====================================================================
# 3. ФУНКЦИИ КЛАССИФИКАЦИИ, СОРТИРОВКИ И ОБРАЩЕНИЯ К API
# =====================================================================
def predict_local_genre(text):
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    cleaned = clean_and_normalize_text(text)
    hash_vector = vectorizer.transform([cleaned])
    return model.predict(hash_vector)[0]

def sort_incoming_documents():
    """Сканирует папку new_documents и раскладывает .txt по папкам-жанрам."""
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
        # Генерируем тестовый файл для проверки работы
        with open(os.path.join(INPUT_FOLDER, "test_space_odyssey.txt"), "w", encoding="utf-8") as f:
            f.write("Космический корабль отправляется в далекую галактику на поиски пришельцев и роботов.")
            
    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.txt')]
    for file_name in files:
        file_path = os.path.join(INPUT_FOLDER, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        genre = predict_local_genre(content)
        target_dir = os.path.join(INPUT_FOLDER, genre)
        os.makedirs(target_dir, exist_ok=True)
        shutil.move(file_path, os.path.join(target_dir, file_name))
        print(f"📁 Документ '{file_name}' успешно перемещен в -> [{genre}]")

def ask_llm_assistant(question):
    """Запрос к LLM модели с жестким System Prompt против галлюцинаций."""
    system_instruction = (
        "Ты — строгий документальный ассистент. Отвечай кратко и строго по фактам. "
        "If you do not know the answer, respond strictly with 'Я не знаю ответ'. "
        "Тебе категорически запрещено выдумывать факты или домысливать информацию."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": question}
            ],
            temperature=0.0  # Полное отключение случайности (фантазии) модели
        )
        return response.choices.message.content
    except Exception as e:
        return f"Ошибка API: {str(e)}"

# =====================================================================
# ТОЧКА ВХОДА
# =====================================================================
if __name__ == "__main__":
    # Если предобученных весов нет — запускаем быстрое обучение (около 20 сек)
    if not os.path.exists(MODEL_PATH):
        train_local_classifier()
        
    # Запуск физического сортировщика файлов
    print("\n[Сортировка входящей документации]:")
    sort_incoming_documents()
    
    # Проверка работы внешнего API с защитой от выдумок
    print("\n[Валидация защиты от галлюцинаций через API]:")
    query = "В каком году изобрели межгалактический телепортатор?"
    print(f"Вопрос: {query}")
    print(f"Ответ ИИ: {ask_llm_assistant(query)}")
