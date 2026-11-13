import os
import re
import csv
import json
import joblib
import logging
import numpy as np
from dotenv import load_dotenv
from langchain_gigachat import GigaChat
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

llm_client = None
knowledge_base = []
ml_vectorizer = None
ml_classifier = None

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
            w = RU_ENDINGS.sub('', w)
        if len(w) >= 3:
            stemmed_words.append(w)
    return " ".join(stemmed_words)

def clean_wiki_garbage(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'mwparseroutput.*?(?=\s[А-ЯЁA-Z]|$)', '', text)
    text = re.sub(r'[a-zA-Z0-9_-]{15,}', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def normalize_macro_genre(raw_genre: str) -> str:
    g = raw_genre.lower()
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

def build_index():
    global llm_client, knowledge_base, ml_vectorizer, ml_classifier
    
    if os.path.exists("hashing_vectorizer.pkl") and os.path.exists("genre_classifier_model.pkl"):
        try:
            ml_vectorizer = joblib.load("hashing_vectorizer.pkl")
            ml_classifier = joblib.load("genre_classifier_model.pkl")
            logging.info("🧠 Усиленные веса LinearSVC (90.43% Top-2) успешно загружены!")
        except Exception as e:
            logging.error(f"Ошибка загрузки весов: {e}")
    else:
        logging.warning("⚠️ Файлы весов (.pkl) не найдены!")

    csv_path = "films_data.csv"
    knowledge_base = []
    
    if not os.path.exists(csv_path):
        logging.error(f"❌ Файл {csv_path} не найден!")
        return

    logging.info("📂 Полная индексация сюжетов с очисткой тегов и сохранением рейтингов...")
    with open(csv_path, mode='r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row.get('title', 'Без названия')
            raw_genre = clean_wiki_garbage(row.get('genre', ''))
            rating_str = str(row.get('imdb_rating', '0')).replace(',', '.').strip()
            
            try:
                rating_num = float(rating_str)
            except ValueError:
                rating_num = 0.0

            summary = clean_wiki_garbage(row.get('summary', ''))
            plot = clean_wiki_garbage(row.get('plot', ''))
            
            full_plot = f"{summary} {plot}".strip()
            if full_plot:
                clean_g = normalize_macro_genre(raw_genre)
                knowledge_base.append({
                    "genre": clean_g,
                    "rating": rating_num,
                    "stemmed_search": stem_russian_text(f"{title} {full_plot}"),
                    "display_text": f"Фильм: «{title}» (IMDb: {rating_num if rating_num > 0 else 'Н/Д'}) | Жанр: {raw_genre}\nСюжет: {full_plot[:800]}"
                })

    logging.info(f"🎯 В оперативную память загружено {len(knowledge_base)} очищенных сюжетов!")

    if os.getenv("GIGACHAT_CREDENTIALS"):
        try:
            from gigachat import GigaChat as DirectGigaChat
            active_model = None
            with DirectGigaChat(credentials=os.getenv("GIGACHAT_CREDENTIALS"), verify_ssl_certs=False) as client:
                models_list = client.get_models()
                for m in models_list.data:
                    if "embed" not in m.id_.lower():
                        active_model = m.id_
                        break
            
            active_model = active_model or "GigaChat"
            llm_client = GigaChat(
                credentials=os.getenv("GIGACHAT_CREDENTIALS"), 
                model=active_model,
                verify_ssl_certs=False,
                temperature=0.2
            )
            logging.info(f"🚀 Подключен генератор: {active_model}")
        except Exception as e:
            logging.error(f"Ошибка подключения к GigaChat: {e}")
def rag_answer(question: str, k: int = 3):
    """Автономный ИИ-агент: сам определяет тип задачи, параметры поиска и отвечает строго по базе"""
    global llm_client, knowledge_base, ml_vectorizer, ml_classifier
    try:
        # Шаг 1: ИИ-маршрутизатор сам разбирает любой вопрос пользователя на параметры
        intent = {"sort": "relevance", "keywords": question, "use_ml": True}
        
        if llm_client:
            router_prompt = [
                SystemMessage(content=(
                    "Ты — поисковый маршрутизатор кинобазы. Проанализируй вопрос пользователя и верни ТОЛЬКО валидный JSON без markdown-обертки:\n"
                    "{\n"
                    '  "sort": "rating_desc" (если просят лучшие/топ/высокий рейтинг), "rating_asc" (если просят худшие/низкий рейтинг), или "relevance" (во всех остальных случаях),\n'
                    '  "keywords": "6-8 ключевых существительных для поиска по сюжету (если упомянут известный фильм, опиши его суть словами)"\n'
                    "}"
                )),
                HumanMessage(content=question)
            ]
            try:
                raw_json = llm_client.invoke(router_prompt).content
                # Очищаем ответ от возможных кавычек markdown
                clean_json = re.sub(r'```json|```', '', raw_json).strip()
                intent = json.loads(clean_json)
                logging.info(f"🧭 Маршрутизатор разобрал запрос: {intent}")
            except Exception:
                pass # Если не удалось распарсить JSON, используем стандартный поиск

        # Шаг 2: Определяем жанр через наши обученные веса (90.43% Top-2)
        search_text = f"{question} {intent.get('keywords', '')}"
        stemmed_q = stem_russian_text(search_text)
        
        top_genres = []
        if ml_vectorizer and ml_classifier:
            vec_q = ml_vectorizer.transform([stemmed_q])
            probs = ml_classifier.predict_proba(vec_q)[0]
            top2_indices = np.argsort(probs)[-2:][::-1]
            top_genres = [str(ml_classifier.classes_[idx]) for idx in top2_indices]
            logging.info(f"🎯 Веса определили жанры: {top_genres}")

        # Выбираем пул фильмов (если вопрос общий без привязки к сюжету — берем всю базу)
        if intent.get("sort") in ["rating_desc", "rating_asc"] and len(question.split()) <= 4:
            search_pool = [item for item in knowledge_base if item["rating"] > 0]
        else:
            search_pool = [item for item in knowledge_base if item["genre"] in top_genres] if top_genres else knowledge_base
            if len(search_pool) < k:
                search_pool = knowledge_base

        # Шаг 3: Выполняем поиск или сортировку по команде маршрутизатора
        sort_mode = intent.get("sort", "relevance")
        
        if sort_mode == "rating_desc":
            rated_pool = [item for item in search_pool if item["rating"] > 0]
            rated_pool.sort(key=lambda x: x["rating"], reverse=True)
            relevant_docs = [item["display_text"] for item in rated_pool[:k]]
            
        elif sort_mode == "rating_asc":
            rated_pool = [item for item in search_pool if item["rating"] > 0]
            rated_pool.sort(key=lambda x: x["rating"])
            relevant_docs = [item["display_text"] for item in rated_pool[:k]]
            
        else:
            stop_roots = {'найд', 'фильм', 'расскаж', 'сюжет', 'про', 'какой', 'котор', 'очен', 'кин', 'ил', 'похож', 'врод', 'тип', 'посовету'}
            keywords = list({w for w in stemmed_q.split() if len(w) >= 3 and w not in stop_roots})
            
            scored_docs = []
            for item in search_pool:
                unique_matches = sum(1 for word in keywords if word in item["stemmed_search"])
                if unique_matches > 0:
                    freq_bonus = sum(item["stemmed_search"].count(word) for word in keywords) * 0.1
                    scored_docs.append((unique_matches + freq_bonus, item["display_text"]))
            
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            relevant_docs = [doc for _, doc in scored_docs[:k]]
            
            if not relevant_docs:
                relevant_docs = [item["display_text"] for item in search_pool[:k]]

        context_text = "\n\n".join(relevant_docs)
        
        # Шаг 4: Финальный ответ строго по найденным документам
        if llm_client:
            messages = [
                SystemMessage(content=(
                    "Ты — умный ИИ-киноэксперт, работающий строго по локальной базе данных. "
                    "Обсуждай и называй ТОЛЬКО те фильмы, которые приведены в разделе 'Выписка из базы данных'. "
                    "Запрещено вспоминать другие фильмы из своей памяти! Подробно ответь на вопрос пользователя, "
                    "опираясь на сюжеты и рейтинги IMDb из выписки."
                )),
                HumanMessage(content=f"Выписка из базы данных:\n{context_text}\n\nВопрос пользователя: {question}")
            ]
            response = llm_client.invoke(messages)
            return response.content, relevant_docs
        else:
            return "Ошибка: API-клиент Сбера не активен.", []
            
    except Exception as e:
        return f"Ошибка RAG-модуля: {str(e)}", []