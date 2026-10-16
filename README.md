# Ассистент по законам Eclipse RP · EclipseLaw

**[Русский](#русский) · [English](#english)**

> 🌐 **Язык интерфейса: только русский** · **Interface language: Russian only**

Ассистент-оверлей для GTA 5 RP (сервер Eclipse): отвечает на вопросы по законам сервера — находит нужные статьи и объясняет. Работает локально на RTX 5080.

An assistant/overlay for GTA 5 RP (Eclipse server): answers questions about the server's laws — finds the relevant articles and explains them. Runs locally on an RTX 5080.

---

## Русский

### Что это
Помощник по внутриигровому законодательству RP-сервера. Задаёте вопрос — он ищет по базе законов (RAG) и отвечает со ссылкой на статьи. Локальная модель через Ollama.

### Возможности
- ⚖️ Ответы по законам сервера (поиск по базе + объяснение)
- 🧠 RAG: модель отвечает строго по документам
- 🪟 Режим оверлея и консольный режим
- 🔧 Разные модели под задачи (напр. крупнее — на суд/прокуратуру)

### Требования
- Windows 10/11, видеокарта (RTX 5080 — эталон)
- Python 3.12, Ollama + модели
- Документы с законами (импортируются в базу)

### Запуск
- `Установить зависимости.bat`
- `Запустить ассистент.bat` (оверлей) или `… (консоль).bat`

---

## English

### What it is
An assistant for an RP server's in-game laws. Ask a question — it searches the law base (RAG) and answers with article references. Local model via Ollama.

### Features
- ⚖️ Answers about server laws (search + explanation)
- 🧠 RAG: the model answers strictly from the documents
- 🪟 Overlay mode and console mode
- 🔧 Different models per task (e.g. a larger one for court/prosecution)

### Requirements
- Windows 10/11, a GPU (RTX 5080 reference)
- Python 3.12, Ollama + models
- Law documents (imported into the base)

### Run
- `Установить зависимости.bat`
- `Запустить ассистент.bat` (overlay) or `… (консоль).bat`

---

### Технологии / Tech
Python · Ollama · RAG (векторный поиск / vector search)

### ⚠️
Личный проект, привязан к конкретному RP-серверу. / Personal project, tied to a specific RP server.
