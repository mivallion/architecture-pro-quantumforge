# Задание 7. Аналитика покрытия базы знаний

## Что реализовано

- Три документа перемещены из активной базы в `Task7/withheld_documents`:
  `Tempest Duke`, `Vitalist`, `Gearling Artificer`.
- Индекс обновлён: сейчас в нём 34 документа и 50 чанков.
- Все запросы Telegram-бота и CLI проходят через `LoggingRagService` и
  записываются в `Task7/logs.jsonl`.
- `Task7/golden_questions.txt` содержит 13 размеченных вопросов: 8 известных и
  5 отсутствующих.
- `app/evaluate_knowledge_base.py` выполняет полный retrieval → Ollama прогон и
  проверяет тип ответа, ожидаемый источник, обязательные и запрещённые факты.
- Итоговый машинный отчёт сохранён в `Task7/evaluation_results.json`, выводы —
  в `Task7/analysis_report.md`, sequence diagram — в `Task7/sequence.puml`.

## Формат лога

Каждая JSONL-запись содержит:

- `timestamp` и полный `query`;
- `result` и `response_length`;
- `chunks_found` и число retrieved chunks;
- `sources` — источники, найденные retrieval;
- `cited_sources` — источники, реально процитированные моделью;
- `successful_answer`, `status` и безопасный `error_type`.

Разделение retrieved и cited sources принципиально: FAISS нашёл правильный
документ для Abyssal Devourer, но генерация всё равно закончилась отказом.

## Запуск

Сначала запустить Ollama:

```bash
OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve
```

Затем из корня репозитория:

```bash
HF_HUB_OFFLINE=1 venv/bin/python -m app.evaluate_knowledge_base --reset-log
```

`--reset-log` нужен только для чистого воспроизводимого эксперимента: он
пересоздаёт `Task7/logs.jsonl`. Без флага новые запросы дописываются в файл.

Тесты evaluator и логирования:

```bash
venv/bin/python -m pytest \
  app/tests/test_evaluate_knowledge_base.py \
  app/tests/test_query_logging.py
```

## Фактический результат

| Проверка | Результат |
|---|---:|
| Общий pass rate | 11/13 (84,62%) |
| Известные темы | 6/8 (75%) |
| Корректные отказы на отсутствующих темах | 5/5 (100%) |
| Retrieval recall ожидаемого документа@4 | 8/8 (100%) |

Две ошибки известных вопросов имеют разные причины:

- по Abyssal Devourer retrieval нашёл документ, но полный ответ отклонён на
  стадии генерации/валидации;
- Chromist ответил с ожидаемым источником, но добавил ложный факт из
  нерелевантного чанка.

Подробный разбор и рекомендации находятся в `Task7/analysis_report.md`.

## Восстановление полной базы

После проверки три файла можно вернуть из `Task7/withheld_documents` в
`Task2/knowledge_base`, затем выполнить:

```bash
venv/bin/python -m app.update_index
```

В текущем состоянии пробелы оставлены намеренно, поскольку это является частью
результата задания 7.
