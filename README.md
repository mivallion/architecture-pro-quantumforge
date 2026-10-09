# QuantumForge RAG Bot

Учебный локальный RAG-бот для проектной работы по software architecture. Бот
ищет сведения в вымышленной базе Aetherfall, формирует подтверждённые ответы со
ссылками на чанки, безопасно отказывается при недостатке данных и доступен через
Telegram.

Сводное решение всех семи заданий находится в
[`Project_template.md`](Project_template.md). Детальные отчёты, реальные логи и
диаграммы лежат в каталогах `Task1`–`Task7`.

## Стек

- Python 3.14;
- `Qwen/Qwen3-Embedding-0.6B`, embeddings размерности 512;
- FAISS `IndexIDMap2(IndexFlatIP)`;
- SQLite для чанков, метаданных и document hashes;
- `qwen3:4b-instruct-2507-q4_K_M` через Ollama;
- `python-telegram-bot`;
- `RecursiveCharacterTextSplitter`;
- Docker и Docker Compose.

Текущий snapshot после искусственных пробелов задания 7 содержит 34 документа
и 50 чанков. Первоначальная база задания 2 состояла из 36 сущностей; три
документа временно сохранены в `Task7/withheld_documents`, а один вредоносный
fixture из задания 5 остаётся в активном индексе для security regression.

## Быстрый запуск через Docker Compose

Требуются Docker Desktop с поддержкой Compose и интернет при первом запуске.
Первый build скачивает embedding-модель, а init-контейнер — Qwen 4B для Ollama,
поэтому потребуется несколько гигабайт диска и некоторое время.

Создайте локальный файл окружения:

```bash
cp .env.example .env
chmod 600 .env
```

Укажите настоящий `TELEGRAM_BOT_TOKEN` в `.env`, затем проверьте и запустите
конфигурацию:

```bash
docker compose config
docker compose up --build
```

Compose запускает три сервиса:

1. `ollama` — локальный Ollama API с persistent model volume;
2. `ollama-model` — one-shot загрузка выбранной Qwen-модели;
3. `bot` — Telegram-бот с read-only mount текущего SQLite/FAISS индекса.

Логи контейнеров:

```bash
docker compose logs -f bot
```

Остановка:

```bash
docker compose down
```

Модели Ollama и query logs остаются в Docker volumes между запусками. Команда
`docker compose down --volumes` удалит эти данные и для обычной остановки не
нужна.

На macOS контейнер Ollama не использует нативное Metal-ускорение хоста, поэтому
полностью контейнерный вариант может отвечать заметно медленнее локального
запуска Ollama. Compose предназначен прежде всего для воспроизводимости и
проверки упаковки.

## Локальный запуск

### 1. Python

```bash
python3.14 -m venv venv
venv/bin/pip install -r app/requirements.txt
venv/bin/pip install -r app/requirements-dev.txt
```

### 2. Ollama

Установите Ollama и в первом терминале запустите сервер:

```bash
OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve
```

Во втором терминале один раз скачайте модель:

```bash
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

### 3. Telegram token

```bash
cp .env.example .env
chmod 600 .env
```

Замените placeholder `TELEGRAM_BOT_TOKEN` значением от BotFather. `.env`
исключён из Git и никогда не копируется в Docker image.

### 4. Запуск бота

Проверить Telegram token:

```bash
venv/bin/python -m app.check_telegram
```

Запустить polling:

```bash
venv/bin/python -m app.bot
```

Одиночный запрос без Telegram:

```bash
venv/bin/python -m app.query_rag "Как призвать Watcher of the Rift?"
```

## Индекс и база знаний

Готовый snapshot уже сохранён в:

- `Task3/index/faiss.index`;
- `Task3/index/metadata.sqlite3`.

Повторная инкрементальная индексация:

```bash
venv/bin/python -m app.build_index
```

Безопасное обновление с lock, retry, JSONL-логом и проверкой согласованности:

```bash
venv/bin/python -m app.update_index
```

Генератор базы задания 2 имеет отдельные зависимости:

```bash
venv/bin/pip install -r Task2/requirements.txt
venv/bin/python -m Task2.build_knowledge_base
```

Для полной регенерации нужен исходный XML dump
`data/terraria_gamepedia_pages_current.xml`. Он намеренно не включён в Git из-за
размера около 107 МБ; готовые очищенные документы и словарь замен включены.

## Проверки и эксперименты

Все unit/integration tests без сетевых обращений:

```bash
venv/bin/python -m pytest -q
```

Retrieval evaluation задания 4:

```bash
venv/bin/python -m app.evaluate_retrieval
```

Prompt-injection эксперимент задания 5, при запущенном Ollama:

```bash
venv/bin/python -m app.evaluate_prompt_injection
```

Golden-set задания 7:

```bash
HF_HUB_OFFLINE=1 venv/bin/python -m app.evaluate_knowledge_base --reset-log
```

Финальный golden-run дал 11/13, retrieval recall ожидаемого документа@4 — 8/8,
правильные отказы на отсутствующих темах — 5/5. Подробный разбор находится в
`Task7/analysis_report.md`.

## Безопасность

- Retrieved documents считаются недоверенными данными.
- Prompt-injection чанки фильтруются до вызова LLM.
- Ответ принимается только с валидными ссылками на переданные источники.
- При неопределённости возвращается «Я не знаю».
- Telegram token передаётся только через `.env` или переменную окружения.
- `.env`, model caches, runtime logs и исходный XML dump исключены из Git.

Fixture со строкой `swordfish` является искусственной canary из задания, а не
настоящим секретом.

## Структура репозитория

| Путь | Назначение |
|---|---|
| `app/` | индексатор, retrieval, RAG, Telegram, security и evaluation |
| `Task1/Solution.md` | выбор моделей, хранилища и инфраструктуры |
| `Task2/` | генератор и уникальная Markdown-база Aetherfall |
| `Task3/` | FAISS/SQLite snapshot и результаты поиска |
| `Task4/` | RAG-описание и проверочные диалоги |
| `Task5/` | prompt-injection эксперимент и демонстрационные логи |
| `Task6/` | ежедневное обновление, cron/launchd и PlantUML |
| `Task7/` | golden set, query logs, quality report и sequence diagram |
| `Dockerfile`, `compose.yaml` | контейнерная упаковка проекта |

## Известные ограничения

- Локальная Qwen 4B рассчитана на одного пользователя и учебную нагрузку.
- Контекст сейчас ограничивается символами, а не токенами; длинный набор
  чанков может приблизиться к `num_ctx=2048`.
- Regex-защита от prompt injection является одним из слоёв, а не полной
  гарантией безопасности.
- FAISS встроен в один процесс и не предоставляет серверную репликацию, ACL и
  distributed update.
- Query logs задания 7 содержат полный текст вопросов и ответов; в production
  необходимы retention policy и маскирование персональных данных.
