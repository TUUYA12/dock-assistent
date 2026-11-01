import os
import csv
import logging
from dotenv import load_dotenv
from langchain_gigachat import GigaChat
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

llm_client = None
knowledge_base = []

def build_index():
    """Мгновенная загрузка текстовой базы и динамическое определение живой модели GigaChat"""
    global llm_client, knowledge_base
    
    csv_path = "films_data.csv"
    knowledge_base = []
    
    if not os.path.exists(csv_path):
        logging.error(f"❌ Файл {csv_path} не найден!")
        return

    logging.info("📂 Загрузка текстовой базы сюжетов...")
    try:
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= 1000: break # Увеличили до 1000 для лучшего охвата сюжетов
                title = row.get('title', row.get('name', ''))
                genre = row.get('genre', '')
                description = row.get('description', row.get('text', ''))
                if description:
                    knowledge_base.append(f"Фильм: {title} | Жанр: {genre} | Сюжет: {description[:600]}")
    except Exception:
        with open(csv_path, mode='r', encoding='cp1251') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= 1000: break
                title = row.get('title', row.get('name', ''))
                genre = row.get('genre', '')
                description = row.get('description', row.get('text', ''))
                if description:
                    knowledge_base.append(f"Фильм: {title} | Жанр: {genre} | Сюжет: {description[:600]}")

    logging.info(f"🎯 Сюжетов загружено: {len(knowledge_base)}")

    # Динамический подбор модели через прямой клиент, чтобы обойти 404 ошибку
    if os.getenv("GIGACHAT_CREDENTIALS"):
        try:
            from gigachat import GigaChat as DirectGigaChat
            active_model = None
            
            # Запрашиваем у Сбера список моделей, доступных именно вашему ключу прямо сейчас
            with DirectGigaChat(credentials=os.getenv("GIGACHAT_CREDENTIALS"), verify_ssl_certs=False) as client:
                models_list = client.get_models()
                for m in models_list.data:
                    # Нам нужна текстовая модель, а не эмбеддинг
                    if "embed" not in m.id_.lower():
                        active_model = m.id_
                        break
            
            # Если Сбер ничего не вернул, ставим стандартный рабочий фолбэк
            if not active_model:
                active_model = "GigaChat"
                
            logging.info(f"🎯 Автоматически выбрана активная модель Сбера: {active_model}")
            
            llm_client = GigaChat(
                credentials=os.getenv("GIGACHAT_CREDENTIALS"), 
                model=active_model,
                verify_ssl_certs=False,
                temperature=0.2
            )
            logging.info("🚀 Успешное подключение к облаку GigaChat API!")
        except Exception as e:
            logging.error(f"Ошибка автоматического определения модели: {e}")

def rag_answer(question: str, k: int = 3):
    """Прямой и умный поиск по сюжету через контекстное окно API ключа"""
    global llm_client, knowledge_base
    try:
        # Ищем ключевые слова вопроса в нашей базе сюжетов (простой и сверхбыстрый конвейер)
        keywords = question.lower().split()
        relevant_docs = []
        
        for doc in knowledge_base:
            # Если сюжет содержит ключевые слова из вопроса — берем его в контекст
            if any(word in doc.lower() for word in keywords if len(word) > 3):
                relevant_docs.append(doc)
            if len(relevant_docs) >= k:
                break
                
        # Если ничего специфичного не нашли, берем первые релевантные записи
        if not relevant_docs:
            relevant_docs = knowledge_base[:k]
            
        context_text = "\n".join(relevant_docs)
        
        if llm_client:
            messages = [
                SystemMessage(content=(
                    "Ты — продвинутый ИИ-аналитик киносюжетов. Отвечай на вопросы пользователя развернуто, "
                    "опираясь исключительно на предоставленный контекст фильмов. Выделяй названия и суть происходящего."
                )),
                HumanMessage(content=f"Выписка из базы сюжетов:\n{context_text}\n\nВопрос: {question}")
            ]
            response = llm_client.invoke(messages)
            return response.content, relevant_docs
        else:
            return "Ошибка: API-клиент Сбера не активен.", []
            
    except Exception as e:
        return f"Ошибка модуля: {str(e)}", []
