"""
Проверка парсера на реальном тексте, скопированном с форума.

Запуск:
    python tools/smoke_parser.py <файл.txt>
    python tools/smoke_parser.py --clipboard
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import Config, setup_logging  # noqa: E402
from core.legal.parser import ForumParser  # noqa: E402

# Контрольные точки, снятые с реального документа «Общие правила проекта».
# Если парсер сломается на них — импорт молча испортит базу.
# Контрольные точки по документам. Ключ — заголовок, который парсер
# обязан извлечь сам. Незнакомый документ проверяется только структурно:
# дубли, мусор форума, ложные санкции, заголовки разделов.
EXPECTED: dict[str, dict] = {
    "Общие правила проекта": {
        "разделов": 7,
        "первый пункт": "1.1",
        "последний пункт": "7.7",
        "пунктов": 98,
        "разделы": {"1": "Общее положение", "6": "Правила RolePlay процесса"},
    },
    "Правила зеленых зон": {
        "разделов": 2,
        "первый пункт": "1.1",
        "последний пункт": "2.4",
        "пунктов": 13,
        "разделы": {"1": "Общее положение", "2": "Исключения"},
    },
}


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    if sys.argv[1] == "--clipboard":
        import subprocess
        raw = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "Get-Clipboard -Raw"],
            capture_output=True, text=True, encoding="utf-8").stdout
        source = "буфер обмена"
    else:
        raw = Path(sys.argv[1]).read_text(encoding="utf-8")
        source = sys.argv[1]

    if not raw.strip():
        print("Пусто: нечего разбирать.")
        return 1

    cfg = Config.load()
    setup_logging(cfg)
    parser = ForumParser.from_config(cfg)
    doc = parser.parse(raw)

    print(f"Источник : {source} ({len(raw)} символов)")
    print(f"Документ : {doc.title}")
    print(f"Разделов : {len(doc.sections)}")
    # Номер раздела не всегда число: у глав он римский («Глава I»).
    def section_order(item: tuple[str, str]) -> tuple[int, object]:
        number = item[0]
        return (0, int(number)) if number.isdigit() else (1, number)

    for number, name in sorted(doc.sections.items(), key=section_order):
        count = sum(1 for r in doc.rules if r.section_number == number)
        print(f"   {number}. {name}  — пунктов: {count}")

    print(f"\nПунктов всего : {len(doc.rules)}")
    print(f"  с санкцией  : {sum(1 for r in doc.rules if r.sanction)}")
    print(f"  с примечан. : {sum(1 for r in doc.rules if r.notes)}")

    failures = 0

    if not doc.rules:
        print("\nПРОВАЛ: не разобрано ни одного пункта")
        return 1

    first, last = doc.rules[0].number, doc.rules[-1].number
    print(f"\nПервый пункт: {first}   Последний: {last}")

    # --- контрольные точки по известному документу -------------------------
    expected = EXPECTED.get(doc.title)
    if expected is None:
        print(f"\n  документ «{doc.title}» незнаком — "
              "проверяю только структуру")
    else:
        print(f"\n  контрольные точки для «{doc.title}»:")
        checks = [
            ("разделов", len(doc.sections), expected["разделов"]),
            ("пунктов", len(doc.rules), expected["пунктов"]),
            ("первый пункт", first, expected["первый пункт"]),
            ("последний пункт", last, expected["последний пункт"]),
        ]
        for name, actual, wanted in checks:
            if actual == wanted:
                print(f"    OK     {name}: {actual}")
            else:
                print(f"    ПРОВАЛ {name}: {actual}, ожидалось {wanted}")
                failures += 1

        # Заголовки разделов не должны подменяться преамбулой: во второй
        # раз картинкой-разделителем закрыт вводный абзац, и его последняя
        # строка легко становится «названием раздела».
        for number, wanted in expected["разделы"].items():
            actual = doc.sections.get(number, "(нет)")
            if actual == wanted:
                print(f"    OK     раздел {number}: «{actual}»")
            else:
                print(f"    ПРОВАЛ раздел {number}: «{actual}», "
                      f"ожидался «{wanted}»")
                failures += 1

    # Номера не должны повторяться: дубль означает, что журнал изменений
    # из второго поста не отсечён.
    seen: dict[str, int] = {}
    for rule in doc.rules:
        seen[rule.number] = seen.get(rule.number, 0) + 1
    duplicates = {n: c for n, c in seen.items() if c > 1}
    if duplicates:
        print(f"  ПРОВАЛ: повторяются пункты {duplicates} — "
              "второй пост (журнал изменений) не отрезан")
        failures += 1
    else:
        print("  дублей номеров нет — OK")

    # Мусор из шапки/подвала форума не должен попасть в пункты.
    garbage = [r.number for r in doc.rules if any(
        marker in r.full_text for marker in
        ("gta5_divider", "Добавить закладку", "Реакции:", "Пожаловаться",
         "LikeЖалоба", "Политика конфиденциальности"))]
    if garbage:
        print(f"  ПРОВАЛ: мусор форума внутри пунктов {garbage}")
        failures += 1
    else:
        print("  мусора форума в пунктах нет — OK")

    # «|» из примеров RP-процесса не должен стать санкцией.
    false_sanctions = [r.number for r in doc.rules
                       if r.sanction in ("Неудачно.", "Удачно.")]
    if false_sanctions:
        print(f"  ПРОВАЛ: пример RP принят за санкцию в {false_sanctions}")
        failures += 1
    else:
        print("  примеры RP за санкцию не приняты — OK")

    if doc.warnings:
        print(f"\nПредупреждения ({len(doc.warnings)}):")
        for warning in doc.warnings:
            print(f"  ! {warning}")

    print("\n--- образцы разбора ---")
    for rule in doc.rules[:2]:
        show(rule)
    for rule in doc.rules:
        if rule.number == "1.7.1":
            show(rule)
        if rule.number == "2.7":        # санкция далеко от номера
            show(rule)
        if rule.number == "6.26":       # был склеен с заголовком спойлера
            show(rule)

    print("Итог:", "парсер работает" if failures == 0
          else f"ПРОВАЛОВ: {failures}")
    return 0 if failures == 0 else 1


def show(rule) -> None:
    print(f"\n[{rule.number}] раздел {rule.section_number} "
          f"«{rule.section_title}»")
    body = rule.text if len(rule.text) <= 200 else rule.text[:197] + "..."
    print(f"  текст   : {body}")
    print(f"  санкция : {rule.sanction or '(нет)'}")
    if rule.notes:
        for note in rule.notes:
            print(f"  примеч. : {note[:120]}")
    print(f"  хэш     : {rule.hash[:16]}")


if __name__ == "__main__":
    raise SystemExit(main())
