import os
import re
import csv
import json
import ast
import time
import operator
from dotenv import load_dotenv
from gigachat import GigaChat as DirectGigaChat
from gigachat.models import Chat, Messages, MessagesRole
from langchain_gigachat import GigaChat
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
import requests
import urllib3
urllib3.disable_warnings()

load_dotenv()

MOVIES_DATA = []
if os.path.exists("films_data.csv"):
    with open("films_data.csv", mode="r", encoding="utf-8", errors="replace") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= 5000:
                break
            try:
                rating = float(str(row.get("imdb_rating", "0")).replace(",", "."))
            except ValueError:
                rating = 0.0
            MOVIES_DATA.append({
                "title": row.get("title", ""),
                "genre": row.get("genre", "").lower(),
                "rating": rating,
                "plot": f"{row.get('summary', '')} {row.get('plot', '')}"[:400]
            })

def tool_search(query: str) -> str:
    """Поиск фильмов в локальной базе по названию или ключевым словам сюжета."""
    q_words = [w.lower() for w in query.split() if len(w) > 2]
    results = []
    for m in MOVIES_DATA:
        text = f"{m['title']} {m['plot']}".lower()
        if any(w in text for w in q_words):
            results.append(f"«{m['title']}» (IMDb: {m['rating']}) — {m['plot'][:180]}...")
        if len(results) >= 3:
            break
    return "\n".join(results) if results else "Фильмы по запросу не найдены."

def tool_calc(expression: str) -> str:
    """Вычислить математическое выражение (например, '6.17 - 5.14')."""
    allowed_ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
    def eval_node(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in allowed_ops:
            return allowed_ops[type(node.op)](eval_node(node.left), eval_node(node.right))
        raise ValueError("Недопустимое выражение")
    try:
        clean_expr = re.sub(r'[^0-9+\-*/().\s]', '', expression)
        res = eval_node(ast.parse(clean_expr, mode='eval').body)
        return str(round(res, 2))
    except Exception as e:
        return f"Ошибка калькулятора: {e}"

def tool_genre_stats(genre: str) -> str:
    """Получить статистику по жанру: количество фильмов и средний рейтинг IMDb."""
    g = genre.lower().strip()
    matched = [m["rating"] for m in MOVIES_DATA if g in m["genre"] and m["rating"] > 0]
    if not matched:
        return f"По жанру '{genre}' фильмов с рейтингом не найдено."
    avg = round(sum(matched) / len(matched), 2)
    return f"Жанр '{genre}': найдено {len(matched)} фильмов, средний рейтинг IMDb = {avg}"

TOOLS = {
    "search": tool_search,
    "calc": tool_calc,
    "stats": tool_genre_stats
}

def run_tool(name: str, arg_value) -> str:
    if name not in TOOLS:
        return f"Ошибка: неизвестный инструмент '{name}'. Доступны: {list(TOOLS.keys())}"
    try:
        if isinstance(arg_value, dict):
            val = next(iter(arg_value.values())) if arg_value else ""
            return TOOLS[name](str(val))
        return TOOLS[name](str(arg_value))
    except Exception as e:
        return f"Ошибка инструмента {name}: {e}"

# Прямой вызов через официальный SDK (не падает на pydantic)
def ask_gigachat_direct(sys_text: str, user_text: str) -> str:
    with DirectGigaChat(credentials=os.getenv("GIGACHAT_CREDENTIALS"), verify_ssl_certs=False) as client:
        token = client.get_token().access_token
        
    resp = requests.post(
        "https://sberbank.ru",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={
            "model": "GigaChat-2",
            "messages": [
                {"role": "system", "content": sys_text},
                {"role": "user", "content": user_text}
            ],
            "temperature": 0.0
        },
        verify=False
    )
    data = resp.json()
    return data["choices"][0]["message"].get("content", "")
llm = GigaChat(
    credentials=os.getenv("GIGACHAT_CREDENTIALS"),
    model="GigaChat-2",
    verify_ssl_certs=False,
    temperature=0.0,
    timeout=60
)

def extract_json(text: str) -> dict:
    clean = re.sub(r'```json|```', '', text).strip()
    start = clean.find('{')
    if start == -1:
        raise ValueError("В ответе модели нет JSON")
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(clean[start:])
    return obj

# =====================================================================
# ПОДХОД 1: Ручной JSON-агент (через прямой SDK без багов pydantic)
# =====================================================================
def run_json_agent(question: str, max_steps: int = 6) -> str:
    print("\n" + "="*60)
    print("🤖 ЗАПУСК ПОДХОДА 1: Ручной JSON-агент")
    print("="*60)
    system_prompt = (
        "Ты кино-аналитик. У тебя есть функции: search (поиск фильмов), stats (рейтинг жанра), calc (калькулятор).\n"
        "Отвечай строго одним JSON-объектом:\n"
        '{"tool": "stats", "arg": "комедия"}\n'
        "Когда все данные собраны, верни:\n"
        '{"tool": "final", "answer": "итоговый ответ"}'
    )
    history = f"Задача: {question}\n"
    
    for step in range(max_steps):
        time.sleep(0.5)
        try:
            raw = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=history)]).content
        except Exception as e:
            print(f"[step {step+1}] Сбой API: {e}")
            continue

        print(f"[step {step+1}] raw={raw[:160].replace(chr(10), ' ')}")
        try:
            parsed = extract_json(raw)
        except Exception as e:
            history += f"Ошибка формата ({e}). Напиши один JSON!\n"
            continue
            
        tool_name = str(parsed.get("tool", parsed.get("action", ""))).strip().lower()
        if tool_name == "final":
            return parsed.get("answer", "Ответ не сформирован")
            
        inp = parsed.get("arg", parsed.get("input", ""))
        result = run_tool(tool_name, inp)
        print(f"[step {step+1}] tool={tool_name} arg='{inp}' -> {result[:120]}")
        history += f"Шаг {step+1}: {tool_name}('{inp}') -> {result}\n"
        
    return "Лимит шагов исчерпан."


# =====================================================================
# ПОДХОД 2: Native Tool Calling (Ручной цикл со схемами)
# =====================================================================
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Поиск фильмов в базе по названию или сюжету.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "stats",
            "description": "Статистика и средний рейтинг IMDb по жанру.",
            "parameters": {"type": "object", "properties": {"genre": {"type": "string"}}, "required": ["genre"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calc",
            "description": "Вычислить математическое выражение.",
            "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}
        }
    }
]

def run_native_agent(question: str, max_steps: int = 6) -> str:
    print("\n" + "="*60)
    print("⚡ ЗАПУСК ПОДХОДА 2: Native Tool Calling (Ручной цикл со схемами)")
    print("="*60)
    llm_with_tools = llm.bind_tools(TOOL_SCHEMAS)
    messages = [
        SystemMessage(content="Ты кино-аналитик. Используй инструменты search, stats и calc для точных ответов."),
        HumanMessage(content=question)
    ]
    
    for step in range(max_steps):
        time.sleep(0.5)
        try:
            response = llm_with_tools.invoke(messages)
        except Exception as e:
            return f"Сбой API на шаге {step+1}: {e}"
            
        messages.append(response)
        
        if not response.tool_calls:
            return response.content
            
        for call in response.tool_calls:
            name = call["name"]
            args = call["args"]
            result = run_tool(name, args)
            print(f"[step {step+1}] tool={name} args={args} -> {result[:120]}")
            messages.append(ToolMessage(tool_call_id=call["id"], content=result))
            
    return "Превышен лимит итераций."

# =====================================================================
# ПОДХОД 3: ФРЕЙМВОРК (LangGraph / LangChain Agent)
# =====================================================================
def run_framework_agent(question: str) -> str:
    print("\n" + "="*60)
    print("🚀 ЗАПУСК ПОДХОДА 3: Фреймворк (LangGraph / LangChain Agent)")
    print("="*60)
    from langgraph.prebuilt import create_react_agent
    
    @tool
    def search_movies(query: str) -> str:
        """Поиск фильмов в базе по названию или сюжету."""
        return tool_search(query)

    @tool
    def genre_statistics(genre: str) -> str:
        """Статистика и средний рейтинг IMDb по жанру."""
        return tool_genre_stats(genre)

    @tool
    def calculator(expression: str) -> str:
        """Вычислить математическое выражение."""
        return tool_calc(expression)

    try:
        agent = create_react_agent(llm, [search_movies, genre_statistics, calculator])
        result = agent.invoke({"messages": [HumanMessage(content=question)]})
        
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"[framework step] tool={tc['name']} args={tc['args']}")
        return result["messages"][-1].content
    except Exception as e:
        return f"Сбой фреймворка: {e}"

if __name__ == "__main__":
    complex_query = (
        "Узнай средний рейтинг фильмов в жанре 'комедия' и средний рейтинг в жанре 'ужасы', "
        "а затем вычисли точную разницу между ними с помощью калькулятора."
    )
    
    ans1 = run_json_agent(complex_query)
    print(f"\n✅ Итог JSON-агента:\n{ans1}")
    
    ans2 = run_native_agent(complex_query)
    print(f"\n✅ Итог Native-агента:\n{ans2}")
    
    ans3 = run_framework_agent(complex_query)
    print(f"\n✅ Итог Фреймворк-агента:\n{ans3}")
