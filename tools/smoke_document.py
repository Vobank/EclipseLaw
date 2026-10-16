"""
Проверка генератора документов (этап 8).

Подаёт надиктовку так, как она приходит с микрофона — разговорную,
сбивчивую, — и смотрит, получится ли из неё документ в структуре
образцов.

Главное, что проверяется: модель не должна переносить в новый документ
имена, паспорта и номера постановлений из образцов. Они принадлежат
другим делам, и такой перенос — фальсификация, а не опечатка.

Запуск: python tools/smoke_document.py
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import Config, setup_logging  # noqa: E402
from core.documents import (TemplateStore,  # noqa: E402
                            redact_invented_facts)
from core.llm.ollama_client import LLMUnavailable, Message, OllamaClient  # noqa: E402
from core.prompts import document_prompt  # noqa: E402

# Надиктовка: как человек говорит, а не как пишут в документе.
DICTATION = """\
Значит так, пятнадцатого сентября приняли обращение от гражданина Игоря
Вольского на сотрудника LSPD с жетоном номер 340. Он пишет что его
задержали возле автомойки, сотрудник не представился вообще, жетон не
показал, сразу надел наручники. Потом повёз в участок и по дороге
превышал скорость. В участке арестовал по статье 12.7 на два года.
Я выпустил постановление DJP номер 3505, запросил кадровый аудит и
видеофиксацию. Структура всё предоставила, сотрудника зовут Марк Свенсон,
паспорт 811204. На видео видно что он правда не представился после
надевания наручников. Первичный обыск тоже не провёл.
"""

# Данные из образцов, которых в надиктовке нет. Появление любого из них
# в готовом документе означает, что модель перенесла чужое дело.
FOREIGN = ["Tasia Pacific", "535420", "Lukas Marlboro", "743210",
           "Tark Blaze", "Tema Demorganzaki", "3496", "3498", "3499"]

# Разделы, без которых документ не документ.
SECTIONS = ["Фигуранты дела", "Ход расследования", "Материалы дела",
            "Предварительное заключение", "Окончательное решение"]


def main() -> int:
    cfg = Config.load()
    setup_logging(cfg)

    store = TemplateStore(cfg.resolve_path("storage.templates_dir"))
    templates = store.load("proceedings")
    print(f"Образцов «{templates.title}»: {len(templates.templates)}")
    for template in templates.templates:
        print(f"   — {template.name}")
    if not templates.templates:
        print("ПРОВАЛ: образцов нет, добавьте через tools/add_template.py")
        return 1

    llm = OllamaClient.from_config(cfg)
    ready, reason = llm.check_ready()
    print(f"LLM: {reason}")
    if not ready:
        return 1

    prompt = document_prompt(
        templates.few_shot(
            limit=cfg.get("documents.samples_in_prompt", 2),
            max_chars=cfg.get("documents.max_sample_chars", 7000),
            like=DICTATION),
        templates.skeleton, DICTATION)
    print(f"Промпт: {len(prompt)} символов\n")

    started = time.monotonic()
    try:
        document = llm.chat([Message("system", prompt)]).strip()
    except LLMUnavailable as exc:
        print(f"ПРОВАЛ: {exc}")
        return 1
    elapsed = time.monotonic() - started

    # Через то же вычёркивание, что и бот: иначе проверка ругается на
    # сырой ответ модели, а пользователь получает уже очищенный.
    document, redacted = redact_invented_facts(document, DICTATION)
    if redacted:
        print(f"Вычеркнуто как не сказанное: {redacted}\n")

    print(f"--- ДОКУМЕНТ ({elapsed:.0f} с, {len(document)} символов) ---")
    print(document[:2600])
    print("--- конец ---\n")

    failures = 0

    found = [s for s in SECTIONS if s.lower() in document.lower()]
    print(f"Разделы на месте: {len(found)} из {len(SECTIONS)} "
          f"({', '.join(found) or 'ни одного'})")
    if len(found) < 4:
        print("  ПРОВАЛ: структура образца не воспроизведена")
        failures += 1

    borrowed = [f for f in FOREIGN if f in document]
    if borrowed:
        print(f"  ПРОВАЛ: перенесены данные из образцов: {borrowed}")
        failures += 1
    else:
        print("Чужих имён и номеров из образцов нет — OK")

    # Факты из надиктовки должны дойти до документа.
    expected = ["Марк Свенсон", "811204", "3505", "340", "12.7"]
    lost = [e for e in expected if e not in document]
    if lost:
        print(f"  ПРОВАЛ: потеряны факты из надиктовки: {lost}")
        failures += 1
    else:
        print("Все факты из надиктовки на месте — OK")

    # Время — процессуально значимый факт: по нему считают сроки.
    # Правдоподобное время, взятое из головы, опаснее явного пробела,
    # потому что его никто не перепроверит.
    times = {t for t in re.findall(r"\b\d{1,2}:\d{2}\b", document)
             if t not in DICTATION}
    if times:
        print(f"  ПРОВАЛ: выдумано время {sorted(times)} — в надиктовке его нет")
        failures += 1
    else:
        print("Выдуманного времени нет — OK")

    gaps = document.count("[НЕ УКАЗАНО]")
    print(f"Помечено незаполненных полей: {gaps}")

    print("\nИтог:", "генератор работает" if failures == 0
          else f"ПРОВАЛОВ: {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
