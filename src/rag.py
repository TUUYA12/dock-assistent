import csv
import logging
import os
import re
from operator import itemgetter
import joblib
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_gigachat import GigaChat

load_dotenv()

llm_client = None
knowledge_base = []
ml_vectorizer = None
ml_classifier = None


def clean_wiki_garbage(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"mwparseroutput.*?(?=\s[А-ЯЁA-Z]|$)", "", text)
    text = re.sub(r"[a-zA-Z0-9_-]{15,}", "", text)
    return re.sub(r"\s+", " ", text).strip()


def build_index():
    global llm_client, knowledge_base, ml_vectorizer, ml_classifier

    if os.path.exists("hashing_vectorizer.pkl") and os.path.exists("genre_classifier_model.pkl"):
        try:
            ml_vectorizer = joblib.load("hashing_vectorizer.pkl")
            ml_classifier = joblib.load("genre_classifier_model.pkl")
        except Exception:
            pass

    knowledge_base.clear()
    if not os.path.exists("films_data.csv"):
        logging.error("Файл films_data.csv не найден!")
        return

    with open("films_data.csv", mode="r", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            title = row.get("title", "Без названия")
            raw_genre = clean_wiki_garbage(row.get("genre", ""))
            rating_str = str(row.get("imdb_rating", "0")).replace(",", ".").strip()
            try:
                rating_num = float(rating_str)
            except ValueError:
                rating_num = 0.0

            summary = clean_wiki_garbage(row.get("summary", ""))
            plot = clean_wiki_garbage(row.get("plot", ""))
            full_plot = f"{summary} {plot}".strip()

            if full_plot:
                display = (
                    f"Фильм: «{title}» (IMDb: {rating_num if rating_num > 0 else 'Н/Д'}) | "
                    f"Жанр: {raw_genre}\nСюжет: {full_plot[:800]}"
                )
                knowledge_base.append({
                    "rating": rating_num,
                    "lower_text": display.lower(),
                    "display_text": display,
                })

    logging.info(f"Загружено {len(knowledge_base)} сюжетов!")

    if os.getenv("GIGACHAT_CREDENTIALS") and not llm_client:
        try:
            from gigachat import GigaChat as DirectGigaChat
            active_model = None
            with DirectGigaChat(credentials=os.getenv("GIGACHAT_CREDENTIALS"), verify_ssl_certs=False) as client:
                for m in client.get_models().data:
                    if "embed" not in m.id_.lower():
                        active_model = m.id_
                        break

            llm_client = GigaChat(
                credentials=os.getenv("GIGACHAT_CREDENTIALS"),
                model=active_model or "GigaChat",
                verify_ssl_certs=False,
                temperature=0.1,
            )
            logging.info(f"Подключен генератор: {active_model}")
        except Exception as e:
            logging.error(f"Ошибка GigaChat: {e}")


def rag_answer(question: str, k: int = 3):
    global llm_client, knowledge_base
    try:
        if not knowledge_base or not llm_client:
            build_index()

        stop_words = {
            "найди", "фильм", "фильмы", "расскажи", "сюжет", "про",
            "какой", "какие", "который", "очень", "кино", "или",
            "похожий", "вроде", "типа", "посоветуй", "есть", "что",
            "чо", "как", "мне",
        }
        raw_words = re.sub(r"[^а-яёa-z0-9\s]", " ", question.lower()).split()
        user_roots = [w[:-2] if len(w) > 4 else w for w in raw_words if len(w) >= 3 and w not in stop_words]

        scored_docs = []
        if user_roots:
            for item in knowledge_base:
                text_lower = item["lower_text"]
                matches = sum(1 for root in user_roots if root in text_lower)
                if matches > 0:
                    freq = sum(text_lower.count(root) for root in user_roots)
                    scored_docs.append({"score": matches * 10 + freq, "text": item["display_text"]})

        scored_docs.sort(key=itemgetter("score"), reverse=True)
        relevant_docs = [doc["text"] for doc in scored_docs[:k]]
        exact_match = len(relevant_docs) > 0

        if not exact_match:
            rated_fallback = [item for item in knowledge_base if item["rating"] > 0]
            rated_fallback.sort(key=itemgetter("rating"), reverse=True)
            relevant_docs = [item["display_text"] for item in rated_fallback[:k]]

        context_text = "\n\n".join(relevant_docs)
        if not llm_client:
            return "Ошибка: API-клиент Сбера не активен.", []

        if not exact_match:
            sys_prompt = (
                "Точных совпадений по запросу пользователя в локальной базе данных НЕТ.\n"
                "Обязательно начни ответ с фразы: 'К сожалению, в нашей базе данных нет фильмов точно по вашему запросу, "
                "но исходя из ситуации и высокого рейтинга IMDb, я могу посоветовать следующие картины:' "
                "и кратко опиши фильмы из выписки ниже."
            )
        else:
            sys_prompt = (
                "Ты — строгий аналитик локальной кинобазы. Отвечай ИСКЛЮЧИТЕЛЬНО по тексту выписки из базы данных ниже.\n"
                "ЗАПРЕЩЕНО вспоминать факты и сцены, которых нет в тексте выписки."
            )

        messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=f"Выписка из базы данных:\n{context_text}\n\nВопрос пользователя: {question}"),
        ]
        return llm_client.invoke(messages).content, relevant_docs

    except Exception as e:
        return f"Ошибка RAG-модуля: {str(e)}", []
