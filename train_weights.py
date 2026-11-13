import os
import re
import csv
import joblib
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import top_k_accuracy_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Быстрый встроенный стеммер (отрезает падежные и глагольные окончания русских слов)
RU_ENDINGS = re.compile(r'(ами|ями|ого|его|ому|ему|ыми|ими|овать|евать|ться|тись|ать|ять|ить|еть|уть|ыть|ов|ев|ей|ам|ям|ах|ях|ом|ем|ой|ей|ий|ый|ая|яя|ое|ее|ые|ие|у|ю|а|я|о|е|ы|и|ь)$')

def stem_russian_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^а-яёa-z\s]', ' ', text)
    words = text.split()
    stemmed_words = []
    for w in words:
        if len(w) > 3:
            # Отрезаем окончание, оставляя корень минимум из 3 букв
            w = RU_ENDINGS.sub('', w)
        if len(w) >= 3:
            stemmed_words.append(w)
    return " ".join(stemmed_words)

def normalize_macro_genre(raw_genre: str) -> str:
    """Группируем в 7 четких, непересекающихся макро-жанров"""
    g = raw_genre.lower()
    # Берем ТОЛЬКО первый (главный) жанр из списка в CSV, чтобы убрать шум второстепенных тегов
    first_tag = re.split(r'[,/|]', g)[0].strip()
    
    if any(k in first_tag for k in ['фантаст', 'sci-fi', 'фэнтези', 'fantasy', 'приключен', 'adventure']):
        return 'фантастика и приключения'
    if any(k in first_tag for k in ['ужас', 'horror', 'триллер', 'thriller', 'детектив', 'криминал', 'crime', 'mystery']):
        return 'триллер, ужасы и детектив'
    if any(k in first_tag for k in ['боевик', 'action', 'военн', 'war', 'вестерн']):
        return 'боевик и военный'
    if any(k in first_tag for k in ['комед', 'comedy']):
        return 'комедия'
    if any(k in first_tag for k in ['мульт', 'анимац', 'animation', 'детск', 'семейн', 'family']):
        return 'анимация и семейный'
    if any(k in first_tag for k in ['документ', 'documentary', 'биограф', 'истор']):
        return 'документальный и история'
    if any(k in first_tag for k in ['драм', 'drama', 'мелодрам', 'romance']):
        return 'драма и мелодрама'
    return 'другое'

def train_boosted_weights():
    csv_path = "films_data.csv"
    if not os.path.exists(csv_path):
        logging.error(f"❌ Файл {csv_path} не найден!")
        return

    logging.info("📂 Чтение датасета, выделение главных тегов и стемминг корней слов...")
    texts = []
    genres = []

    with open(csv_path, mode='r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row.get('title', '')
            raw_genre = row.get('genre', '')
            summary = row.get('summary', '')
            plot = row.get('plot', '')
            
            # Берем только записи с полноценным сюжетом (от 80 символов), чтобы исключить пустышки
            full_description = f"{summary} {plot}".strip()
            if raw_genre and len(full_description) > 80:
                clean_g = normalize_macro_genre(raw_genre)
                if clean_g != 'другое':
                    # Усиливаем вес названия фильма (повторяем 2 раза)
                    raw_text = f"{title} {title} {full_description}"
                    stemmed_text = stem_russian_text(raw_text)
                    if len(stemmed_text) > 30:
                        texts.append(stemmed_text)
                        genres.append(clean_g)

    logging.info(f"✅ Подготовлено {len(texts)} эталонных сюжетов. Макро-жанров: {len(set(genres))}")

    X_train, X_test, y_train, y_test = train_test_split(
        texts, genres, test_size=0.15, random_state=42, stratify=genres
    )

    logging.info("🧠 Глубокая векторизация корней слов и словосочетаний (TF-IDF, 100k признаков)...")
    vectorizer = TfidfVectorizer(
        max_features=100000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.85,         # Отсекаем слова-паразиты, которые есть в 85% всех фильмов (например, "фильм", "год", "жизнь")
        sublinear_tf=True
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    logging.info("⚡ Обучение откалиброванного метода опорных векторов (LinearSVC)...")
    base_svc = LinearSVC(C=0.6, class_weight='balanced', random_state=42)
    classifier = CalibratedClassifierCV(base_svc, cv=3) # Калибровка для получения точных вероятностей
    
    classifier.fit(X_train_vec, y_train)

    # Считаем строгую точность (Top-1) и точность попадания в 2 главных жанра (Top-2)
    acc_top1 = classifier.score(X_test_vec, y_test)
    probs = classifier.predict_proba(X_test_vec)
    acc_top2 = top_k_accuracy_score(y_test, probs, k=2, labels=classifier.classes_)

    logging.info(f"🎯 Строгая точность (Top-1): {acc_top1 * 100:.2f}%")
    logging.info(f"🚀 Точность попадания в 2 главных жанра фильма (Top-2): {acc_top2 * 100:.2f}%")

    joblib.dump(vectorizer, "hashing_vectorizer.pkl")
    joblib.dump(classifier, "genre_classifier_model.pkl")
    logging.info("💾 Усиленные веса сохранены!")

if __name__ == "__main__":
    train_boosted_weights()
