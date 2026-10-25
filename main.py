import os
import time
import logging
import joblib
import re
import csv
from dotenv import load_dotenv
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from langchain_gigachat import GigaChat
from langchain_core.messages import SystemMessage, HumanMessage
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Загружаем переменные окружения из файла .env
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("history.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

MODEL_PATH = "genre_classifier_model.pkl"       
VECTORIZER_PATH = "hashing_vectorizer.pkl"       

model = None
vectorizer = None

# Базовые стоп-слова для исключения мусорных токенов
STOP_WORDS = {
    'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 
    'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 
    'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от', 'меня', 'еще'
}
def collapse_genres(raw_genre):
    """Преобразует грязные жанры в 7 базовых категорий по ТЗ."""
    if not raw_genre: 
        return 'драма и мелодрама'
    raw_genre = str(raw_genre).lower()
    
    if any(w in raw_genre for w in ['фантаст', 'фэнтези', 'космос', 'киберпанк']): 
        return 'фантастика и фэнтези'
    elif any(w in raw_genre for w in ['ужас', 'хоррор', 'триллер', 'саспенс']): 
        return 'ужасы и триллеры'
    elif any(w in raw_genre for w in ['комеди', 'сатир', 'юмор']): 
        return 'комедия'
    elif any(w in raw_genre for w in ['военн', 'война', 'арми', 'полк', 'сражен', 'истори', 'биограф']): 
        return 'история и война'
    elif any(w in raw_genre for w in ['боевик', 'экшен', 'детектив', 'кримин', 'преступ', 'спецназ']): 
        return 'боевик и детектив'
    elif any(w in raw_genre for w in ['документаль', 'хроника', 'научн']): 
        return 'документальный'
    elif any(w in raw_genre for w in ['драма', 'мелодрама', 'романти']): 
        return 'драма и мелодрама'
        
    return 'драма и мелодрама'

def clean_and_normalize_text(text):
    """Очищает спецсимволы и выполняет стемминг русского языка."""
    if not isinstance(text, str): 
        return ""
    text = text.lower()
    text = re.sub(r'[^а-яёa-z0-9\s]', ' ', text)
    words = text.split()
    
    cleaned_tokens = []
    for word in words:
        if word not in STOP_WORDS and len(word) > 2:
            # Срезание флексий и окончаний для нормализации слов
            word = re.sub(r'(ами|ями|ов|ев|ах|ях|ых|их|ый|ий|ая|ое|ее|ам|ям|ом|ем|у|ю|а|я|о|е|ы|и|й|ь)$', '', word)
            cleaned_tokens.append(word)
            
    return " ".join(cleaned_tokens)
def train_model_local_safe():
    """Потоковое чтение датасета для абсолютной защиты оперативной памяти."""
    global model, vectorizer
    csv_path = "films_data.csv"
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Критическая ошибка: Файл датасета {csv_path} не найден в корне проекта!")
        
    logging.info("Обучение весов отсутствует. Запуск безопасного переобучения (20 000 строк)...")
    
    # HashingVectorizer защищает память от переполнения
    vectorizer = HashingVectorizer(ngram_range=(1, 2), n_features=2**18, alternate_sign=False)
    model = SGDClassifier(loss='log_loss', class_weight='balanced', random_state=42)
    
    texts = []
    genres = []
    
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            plot = row.get('plot', '').strip()
            raw_genre = row.get('genre', '').strip()
            
            if plot and raw_genre:
                cleaned_plot = clean_and_normalize_text(plot)
                if len(cleaned_plot) >= 50:
                    texts.append(cleaned_plot)
                    genres.append(collapse_genres(raw_genre))
            
            if i >= 20000:
                break
                
    logging.info(f"Выборка нормализована ({len(texts)} строк). Сборка разреженной матрицы признаков...")
    X = vectorizer.transform(texts)
    
    logging.info("Финальная подгонка весов линейного классификатора SGD...")
    model.fit(X, genres)
    
    # Сохраняем скомпилированные веса на диск
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    logging.info("🎉 Локальное обучение успешно завершено! Веса .pkl сохранены на диск.")

# Загрузка готовых весов или автоматический пересчет при их отсутствии
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    logging.info("Локальная модель ИИ и векторизатор успешно загружены из файлов весов.")
except Exception:
    logging.warning("Файлы весов повреждены или не найдены. Переходим к автоматической сборке...")
    try:
        train_model_local_safe()
    except Exception as e:
        logging.error(f"Не удалось инициализировать классификатор: {str(e)}")
def ask_gigachat_assistant(text_content, question):
    """Отправляет запрос в GigaChat, автоматически определяя доступную модель через API Сбера."""
    if not os.getenv("GIGACHAT_CREDENTIALS"):
        return "Ошибка: Ключ GIGACHAT_CREDENTIALS не найден в .env!"

    try:
        # 1. Сначала инициализируем базовый клиент для динамического опроса сервера Сбера
        from gigachat import GigaChat as DirectGigaChat
        
        active_model = None
        with DirectGigaChat(credentials=os.getenv("GIGACHAT_CREDENTIALS"), verify_ssl_certs=False) as client:
            models_list = client.get_models()
            # Ищем первую попавшуюся модель, которая не является эмбеддингом
            for m in models_list.data:
                if "embed" not in m.id_.lower():
                    active_model = m.id_
                    break
        
        if not active_model:
            # Если автопоиск не сработал, ставим железный современный дефолт для физлиц
            active_model = "GigaChat-2-Lite"
            
        logging.info(f"🎯 Автоматически определена рабочая модель Сбера: {active_model}")

        # 2. Инициализируем LangChain с актуальным именем модели
        llm = GigaChat(
            credentials=os.getenv("GIGACHAT_CREDENTIALS"), 
            model=active_model,
            verify_ssl_certs=False,
            temperature=0.0  # Жесткая защита от галлюцинаций по ТЗ
        )
        
        messages = [
            SystemMessage(content="Ты — документальный ИИ-ассистент. Отвечай строго по тексту предоставленного документа. Если в тексте документа нет ответа на вопрос, ты обязан четко написать: 'Я не знаю ответ'."),
            HumanMessage(content=f"Контекст документа:\n{text_content}\n\nВопрос: {question}")
        ]
        
        response = llm.invoke(messages)
        return response.content
        
    except Exception as e:
        logging.error(f"Критический сбой автопоиска моделей или запроса к GigaChat API: {str(e)}")
        return f"Ошибка обработки запроса в LLM: {str(e)}"



def clean_and_classify(file_path):
    """Классифицирует файл локальной моделью и отправляет запрос в GigaChat."""
    global model, vectorizer
    time.sleep(1.0)  # Пауза для Windows, чтобы снять блокировку с нового файла
    
    if model is None or vectorizer is None:
        logging.error("Классификация невозможна: модель не инициализирована.")
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
    except Exception as file_err:
        logging.error(f"Не удалось прочитать файл {file_path}: {str(file_err)}")
        return
    
    cleaned_text = clean_and_normalize_text(raw_text) 
    if len(cleaned_text) < 50:
        logging.warning(f"Документ {file_path} слишком короткий после очистки. Пропуск.")
        return
        
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)
    predicted_genre = str(prediction[0]).strip()
    
    # Физическое перемещение файла на диске в подпапку жанра
    base_dir = os.path.dirname(file_path)
    target_dir = os.path.join(base_dir, predicted_genre)
    os.makedirs(target_dir, exist_ok=True)
    
    filename = os.path.basename(file_path)
    target_path = os.path.join(target_dir, filename)
    
    try:
        os.rename(file_path, target_path)
        logging.info(f"📁 Документ '{filename}' определен как '{predicted_genre}' и успешно перемещен.")
        
        # --- ТЕСТ ЦЕПОЧКИ: АВТОМАТИЧЕСКИЙ АНАЛИЗ ЧЕРЕЗ GIGACHAT ---
        test_question = "Кто главные герои этого сюжета и какие звания они имеют?"
        logging.info(f"Отправка контекста файла '{filename}' в GigaChat...")
        ai_answer = ask_gigachat_assistant(raw_text, test_question)
        logging.info(f"💬 Ответ ИИ-Ассистента:\n{ai_answer}\n")
        
    except Exception as e:
        logging.error(f"Сбой обработки файла '{filename}': {str(e)}")

class NewDocumentHandler(FileSystemEventHandler):
    """Слушатель событий файловой системы."""
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.txt'):
            logging.info(f"Обнаружен новый текстовый документ: {event.src_path}")
            clean_and_classify(event.src_path)

def start_monitoring():
    """Запуск бесконечного цикла фонового мониторинга директории."""
    path_to_watch = "./new_documents"
    os.makedirs(path_to_watch, exist_ok=True)
    
    event_handler = NewDocumentHandler()
    observer = Observer()
    observer.schedule(event_handler, path=path_to_watch, recursive=False)
    
    observer.start()
    logging.info(f"🚀 Фоновый ИИ-агент запущен. Ожидание файлов в: {path_to_watch}")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Мониторинг остановлен пользователем.")
        observer.stop()
    observer.join()

if __name__ == "__main__":
    start_monitoring()
