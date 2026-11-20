import os
import sys
from pydantic import BaseModel

# Полная изоляция процесса Python от прокси-окружения Windows
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["NO_PROXY"] = "localhost,127.0.0.1,api.gigachat.ru,ngw.devices.sberbank.ru"
os.environ["no_proxy"] = "localhost,127.0.0.1,api.gigachat.ru,ngw.devices.sberbank.ru"

# Защита для GigaChat SDK
os.environ["GIGACHAT_DISABLE_SSL"] = "True"

import logging
from fastapi import FastAPI, HTTPException
from src.rag import build_index, rag_answer

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

app = FastAPI(title="Единый RAG-Агент API")

@app.on_event("startup")
def startup_event():
    logging.info("🚀 Старт инициализации агента. Сборка FAISS...")
    build_index()

@app.post("/ask")
def ask(q: dict):
    try:
        question = q.get("question", "")
        k = q.get("k", 1)
        
        if not question:
            raise HTTPException(status_code=400, detail="Параметр 'question' не может быть пустым")
            
        answer, sources = rag_answer(question=question, k=k)
        return {"answer": answer, "sources": sources}
        
    except Exception as e:
        logging.error(f"Ошибка на эндпоинте /ask: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
import re
from operator import itemgetter
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage
import src.rag as rag_module

class AgentRequest(BaseModel):
    question: str
    mode: str

@app.post("/agent")
def run_custom_agent(req: AgentRequest):
    if not rag_module.knowledge_base or not rag_module.llm_client:
        rag_module.build_index()

    kb = rag_module.knowledge_base
    client = rag_module.llm_client
    steps = []
    q_lower = req.question.lower()

    # АГЕНТ 2: Аналитик (считает реальную статистику по CSV-базе)
    if "Аналитик" in req.mode:
        steps.append("🔍 Шаг 1: Сканирование базы данных для сбора статистики...")
        rated_films = [item for item in kb if item["rating"] > 0]
        avg_total = round(sum(x["rating"] for x in rated_films) / len(rated_films), 2) if rated_films else 0
        steps.append(f"📊 Шаг 2: Найдено фильмов с рейтингом IMDb: {len(rated_films)}. Средний рейтинг по базе: {avg_total}")

        words = [w[:-2] if len(w) > 4 else w for w in re.sub(r"[^а-яёa-z0-9\s]", " ", q_lower).split() if len(w) >= 4]
        matched = [x for x in rated_films if any(w in x["lower_text"] for w in words)]
        
        if matched:
            avg_sub = round(sum(x["rating"] for x in matched) / len(matched), 2)
            best = max(matched, key=itemgetter("rating"))
            worst = min(matched, key=itemgetter("rating"))
            steps.append(f"🧮 Шаг 3: По выборке ({len(matched)} фильмов) средний рейтинг = {avg_sub}")
            stats_context = (
                f"Всего фильмов в базе: {len(kb)} (с рейтингом: {len(rated_films)}, средний по базе: {avg_total}).\n"
                f"По запросу найдено фильмов: {len(matched)}, средний рейтинг: {avg_sub}.\n"
                f"Лучший фильм выборки:\n{best['display_text']}\n\n"
                f"Худший фильм выборки:\n{worst['display_text']}"
            )
        else:
            steps.append("⚠️ Шаг 3: Узкая выборка пуста, использую общую статистику базы.")
            stats_context = f"Всего фильмов в базе: {len(kb)}, с рейтингом IMDb: {len(rated_films)}, средний рейтинг: {avg_total}."

        messages = [
            SystemMessage(content="Ты — Агент-Аналитик кинобазы. Отвечай строго с опорой на переданные расчёты и цифры. Если точных данных по узкому запросу нет — честно скажи об этом и приведи общую статистику."),
            HumanMessage(content=f"Результаты расчёта:\n{stats_context}\n\nВопрос: {req.question}")
        ]
        answer = client.invoke(messages).content
        return {"answer": answer, "steps": steps}

    # АГЕНТ 3: Критик-Подборщик (сравнение и выбор под ситуацию)
    else:
        steps.append("🎭 Шаг 1: Поиск кандидатов для сравнения и анализа...")
        words = [w[:-2] if len(w) > 4 else w for w in re.sub(r"[^а-яёa-z0-9\s]", " ", q_lower).split() if len(w) >= 4]
        scored = []
        for item in kb:
            score = sum(1 for w in words if w in item["lower_text"])
            if score > 0:
                scored.append({"score": score, "rating": item["rating"], "text": item["display_text"]})
        
        scored.sort(key=itemgetter("score", "rating"), reverse=True)
        candidates = [x["text"] for x in scored[:4]]
        
        if not candidates:
            steps.append("⚠️ Шаг 2: Точных совпадений не найдено, беру топ фильмов по рейтингу IMDb.")
            top_rated = sorted([x for x in kb if x["rating"] > 0], key=itemgetter("rating"), reverse=True)[:3]
            candidates = [x["display_text"] for x in top_rated]
            prefix = "В базе нет точных совпадений по вашему описанию, но как кинокритик я рекомендую эти высокооценённые картины:\n"
        else:
            steps.append(f"⚖️ Шаг 2: Отобрано {len(candidates)} фильма(ов) для сравнительного разбора.")
            prefix = ""

        messages = [
            SystemMessage(content=(
                "Ты — Агент-Кинокритик. Сравни предложенные фильмы из выписки по атмосфере, сюжету и рейтингу IMDb, "
                "и дай чёткую рекомендацию, какой из них лучше выбрать под запрос пользователя. Не выдумывай факты вне текста."
            )),
            HumanMessage(content=f"{prefix}Кандидаты из базы:\n" + "\n\n".join(candidates) + f"\n\nЗапрос пользователя: {req.question}")
        ]
        answer = client.invoke(messages).content
        return {"answer": answer, "steps": steps + candidates}
