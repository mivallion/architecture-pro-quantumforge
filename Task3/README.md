# Задание 3. Векторный индекс базы знаний

> Числа ниже фиксируют результат задания 3 до последующих экспериментов. В
> задании 5 в индекс добавлен prompt-injection fixture, а в задании 7 три
> документа намеренно исключены для проверки пробелов. Поэтому текущий snapshot
> содержит 34 документа и 50 чанков; исходный результат задания 3 — 36
> документов и 52 чанка.

## Результат

База знаний из задания 2 преобразована в локальный векторный индекс. FAISS
используется для поиска, а SQLite хранит тексты чанков, их источники, позиции и
эмбеддинги. Индексатор поддерживает повторные запуски: неизменившиеся документы
определяются по SHA-256 и не кодируются повторно.

Созданные артефакты:

- `Task3/index/faiss.index` — FAISS-индекс;
- `Task3/index/metadata.sqlite3` — документы, чанки, метаданные и эмбеддинги;
- `app/build_index.py` — точка запуска индексации;
- `app/search_index.py` — пример поиска по созданному индексу.

## Модель эмбеддингов

- Модель: [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- Библиотека: `sentence-transformers==5.7.0`
- Используемая размерность: 512
- Эмбеддинги документов: `encode_document`
- Эмбеддинги запросов: `encode_query`
- Нормализация: L2-нормализация через `normalize_embeddings=True`

Нормализованные векторы сохраняются в `IndexIDMap2(IndexFlatIP)`. Поэтому
скалярное произведение в FAISS эквивалентно cosine similarity. Для текущего
небольшого корпуса используется точный поиск без аппроксимации.

## База знаний и чанкинг

- Источник: `Task2/knowledge_base`
- Документов: 36 Markdown-файлов
- Разбиение: `RecursiveCharacterTextSplitter`
- Размер чанка: до 500 токенов
- Перекрытие: 50 токенов
- Чанков в итоговом индексе: 52

Размеры считаются токенизатором выбранной Qwen-модели. Для каждого чанка в
SQLite сохраняются:

- идентификатор в FAISS;
- путь к исходному файлу;
- заголовок документа;
- порядковый номер чанка;
- начальная позиция в исходном тексте;
- полный текст чанка;
- эмбеддинг.

## Построение индекса

Команды выполняются из корня репозитория:

```bash
venv/bin/pip install -r app/requirements.txt
venv/bin/python -m app.build_index
```

Полная генерация на MacBook Air M1, 8 ГБ заняла **68,94 секунды**. Результат
полного запуска:

```text
Indexed 36/36 documents into 52 chunks in 68.94s (0 unchanged, 0 deleted).
```

При повторном запуске все 36 документов были распознаны как неизменившиеся;
работа самого индексатора без загрузки модели заняла менее 0,01 секунды:

```text
Indexed 0/36 documents into 52 chunks in 0.00s (36 unchanged, 0 deleted).
```

## Примеры поиска

Поиск можно выполнить командой:

```bash
venv/bin/python -m app.search_index "Кто лечит раненых поселенцев?"
```

Первые три результата:

| Место | Score | Документ | Чанк |
|---:|---:|---|---|
| 1 | 0,352 | Vitalist | `21-vitalist.md#0` |
| 2 | 0,289 | Pathkeeper | `19-pathkeeper.md#0` |
| 3 | 0,285 | Quartermaster | `20-quartermaster.md#0` |

Фрагмент найденного чанка:

```text
# Vitalist

The Vitalist is an resident that will spawn once the following criteria have
been met [...] The Vitalist will heal the wayfarer's health and cure them of
any active debuffs at the cost of Coins.
```

Второй проверочный запрос:

```bash
venv/bin/python -m app.search_index "Как призвать Watcher of the Rift?"
```

Первые три результата:

| Место | Score | Документ | Чанк |
|---:|---:|---|---|
| 1 | 0,698 | Watcher of the Rift | `02-watcher-of-the-rift.md#0` |
| 2 | 0,451 | Mirrorborn Pair | `10-mirrorborn-pair.md#0` |
| 3 | 0,442 | Verdant Oracle | `26-verdant-oracle.md#0` |

Фрагмент найденного чанка:

```text
## Spawn

The Watcher of the Rift can be summoned manually using a Rift Lens at Dark
Cycle (beginning at 7:30PM in-game time).
```

Оба запроса возвращают релевантный документ на первом месте. В выдаче также
есть полный текст чанка, его источник, номер и начальная позиция — эти данные
можно использовать для цитирования в API бота.
