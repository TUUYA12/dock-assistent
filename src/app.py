import os
import sys

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
