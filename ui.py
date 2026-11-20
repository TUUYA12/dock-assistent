import streamlit as st
import requests

st.set_page_config(page_title="AI Киноэксперт", page_icon="🎬", layout="wide")

# Боковая панель с переключением агентов
with st.sidebar:
    st.header("🤖 Выбор режима агента")
    agent_mode = st.radio(
        "Выберите активного агента:",
        [
            "🎬 1. RAG-Киноэксперт (Поиск по сюжетам)",
            "📊 2. Агент-Аналитик (Цифры и статистика)",
            "⚖️ 3. Агент-Критик (Сравнение и выбор)",
        ],
    )


    st.markdown("---")
    if "RAG" in agent_mode:
        k_docs = st.slider("Количество фильмов в выборке (k)", 1, 10, 3)
        st.caption(
            "Ищет по сюжетам. Если точного ответа в базе нет — честно скажет и"
            " предложит топ по рейтингу IMDb."
        )
    else:
        analyst_type = st.selectbox(
            "Архитектура агента:",
            ["LangGraph (ReAct)", "Native Tool Calling", "Ручной JSON-цикл"],
        )
        st.caption(
            "Использует инструменты (search, stats, calc) для подсчёта средних"
            " рейтингов и сравнения жанров."
        )

    st.markdown("---")
    if st.button("🗑️ Очистить историю чата"):
        st.session_state.messages = []
        st.rerun()

st.title(agent_mode)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Источники / Шаги работы агента"):
                for src in msg["sources"]:
                    st.text(src)

placeholder = (
    "Например: фильм про космических пиратов..."
    if "RAG" in agent_mode
    else "Например: сравни средний рейтинг комедий и ужасов..."
)

if prompt := st.chat_input(placeholder):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Агент думает..."):
            try:
                session = requests.Session()
                session.trust_env = False

                if "RAG" in agent_mode:
                    resp = session.post(
                        "http://localhost:8000/ask",
                        json={"question": prompt, "k": k_docs},
                        timeout=60,
                    )
                    data = resp.json()
                    answer = data.get("answer", "Нет ответа")
                    sources = data.get("sources", [])
                else:
                    # Вызов эндпоинта аналитика (если добавите /agent в src/app.py)
                    resp = session.post(
                        "http://localhost:8000/agent",
                        json={"question": prompt, "mode": analyst_type},
                        timeout=90,
                    )
                    if resp.status_code == 404:
                        answer = (
                            "⚠️ Эндпоинт `/agent` ещё не подключён в `src/app.py`."
                            " Сейчас доступен только режим RAG-Киноэксперт."
                        )
                        sources = []
                    else:
                        data = resp.json()
                        answer = data.get("answer", "Нет ответа")
                        sources = data.get("steps", [])

                st.markdown(answer)
                if sources:
                    with st.expander("📚 Источники / Шаги работы агента"):
                        for src in sources:
                            st.text(src)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })
            except Exception as e:
                st.error(f"Ошибка подключения к API: {e}")
