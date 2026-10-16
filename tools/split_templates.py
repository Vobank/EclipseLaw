"""
Разбирает один вставленный файл на отдельные образцы документов.

Судья присылает подборку одним куском — из Google Docs удобнее всего
скопировать её целиком, а не двадцать девять раз по одному. Здесь этот
кусок режется на документы и раскладывается по набору образцов.

Границу документа скрипт ищет сам: берёт строки, которые повторяются
по всему файлу и выглядят как заголовок (короткие, без точки в конце),
и предлагает ту, что делит текст на самые ровные части. Угадывать
вслепую он не станет — сначала показывает разбивку, и только по
команде применяет.

Запуск:
    python tools/split_templates.py <файл>              показать разбивку
    python tools/split_templates.py <файл> --marker "В Верховный Суд"
    python tools/split_templates.py <файл> --apply      добавить образцы
    python tools/split_templates.py <файл> --apply --proceedings
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import Config, setup_logging  # noqa: E402
from core.documents import KINDS, TemplateStore  # noqa: E402

# Документ короче этого — скорее обрывок или шапка, чем образец.
MIN_DOCUMENT = 200


def argument(flag: str, default=None):
    if flag not in sys.argv:
        return default
    index = sys.argv.index(flag)
    return sys.argv[index + 1] if index + 1 < len(sys.argv) else default


def looks_like_heading(line: str) -> bool:
    line = line.strip()
    if not 3 < len(line) < 90:
        return False
    if line.endswith((".", ",", ";", ":")):
        return False
    # Заголовок документа почти всегда начинается с заглавной или цифры.
    return bool(re.match(r"[А-ЯЁA-Z0-9«\"]", line))


def candidates(lines: list[str]) -> list[tuple[int, str]]:
    """Строки-кандидаты в границу: часто повторяются и похожи на заголовок."""
    counter = Counter(line.strip() for line in lines if looks_like_heading(line))
    return [(count, text) for text, count in counter.most_common(12)
            if count >= 2]


def split_by(lines: list[str], marker: str, regex: bool = False) -> list[str]:
    """Режет по строке-границе. regex=True — граница задана шаблоном.

    Шаблон нужен чаще, чем кажется: одна и та же шапка приезжает в двух
    написаниях — «ОКРУЖНОЙ СУД ШТАТА САН-АНДРЕАС» и «...ШТАТА SAN
    ANDREAS». По точной строке такой файл делится лишь наполовину, и
    восемь документов молча слипаются с соседними.
    """
    pattern = re.compile(marker) if regex else None

    def is_boundary(line: str) -> bool:
        text = line.strip()
        return bool(pattern.search(text)) if pattern else text == marker

    chunks: list[list[str]] = []
    for line in lines:
        if is_boundary(line) and chunks and any(chunks[-1]):
            chunks.append([])
        if not chunks:
            chunks.append([])
        chunks[-1].append(line)
    return ["\n".join(c).strip() for c in chunks if "\n".join(c).strip()]


def starts_documents(lines: list[str], marker: str) -> float:
    """Доля кусков, где граница — первая же содержательная строка.

    Так шапка документа отличается от просто частого заголовка.
    """
    chunks = split_by(lines, marker)
    if not chunks:
        return 0.0
    first = sum(
        1 for chunk in chunks
        if next((x.strip() for x in chunk.split("\n") if x.strip()), "") == marker)
    return first / len(chunks)


def main() -> int:
    paths = [a for a in sys.argv[1:] if not a.startswith("-")]
    marker_arg = argument("--marker")
    if marker_arg in paths:
        paths.remove(marker_arg)
    if not paths:
        print(__doc__)
        return 2

    source = Path(paths[0])
    if not source.exists():
        print(f"{source}: файл не найден")
        return 1

    text = source.read_text(encoding="utf-8", errors="replace")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    print(f"Файл: {source}  ({len(text)} символов, {len(lines)} строк)\n")

    marker = marker_arg or argument("--regex")
    if marker in paths:
        paths.remove(marker)
    if not marker:
        found = candidates(lines)
        if not found:
            print("Повторяющихся заголовков не нашлось. Похоже, документы "
                  "не отбиты одинаковой строкой.")
            print("Укажите границу вручную: --marker \"текст строки\"")
            print("\nПервые строки файла — вдруг граница видна глазом:")
            for line in [x for x in lines if x.strip()][:15]:
                print(f"   |{line[:88]}")
            return 1
        print("Похожие на границу строки:")
        scored = []
        for count, line in found:
            share = starts_documents(lines, line)
            scored.append((share, count, line))
            print(f"   {count:3}x  начинает документ в {share * 100:3.0f}% "
                  f"случаев  {line[:50]}")
        # Границей берём не самую частую строку, а ту, что чаще всего
        # стоит ПЕРВОЙ в своём куске: это и есть начало документа.
        # В судебной подборке «ОПРЕДЕЛЕНИЕ» встречается чаще всего, но
        # идёт второй строкой, после названия суда, и деление по нему
        # отрезало бы у каждого документа шапку.
        scored.sort(key=lambda x: (-x[0], -x[1]))
        marker = scored[0][2]
        print(f"\nБеру ту, что начинает документы: «{marker}»")

        # Та же шапка в другом написании поделит файл лишь наполовину,
        # и часть документов молча слипнётся с соседями. Предлагаем
        # шаблон, покрывающий оба написания.
        head = " ".join(marker.split()[:3])
        family = [l for _, _, l in scored if l.startswith(head) and l != marker]
        if family:
            print(f"Та же шапка иначе: {'; '.join(x[:40] for x in family)}")
            print(f"Учесть обе: --regex \"^{re.escape(head)}\"")
        print()

    chunks = split_by(lines, marker, regex=bool(argument("--regex")))
    good = [c for c in chunks if len(c) >= MIN_DOCUMENT]
    small = len(chunks) - len(good)

    print(f"Получилось кусков: {len(chunks)}, из них годных: {len(good)}")
    if small:
        print(f"Отброшено как слишком короткие (<{MIN_DOCUMENT} симв.): {small}")
    print()
    for index, chunk in enumerate(good, 1):
        head = next((x.strip() for x in chunk.split("\n")[:4] if x.strip()), "")
        print(f"  {index:2}. {len(chunk):5} симв.  {head[:66]}")

    if "--apply" not in sys.argv:
        print("\nЭто черновой прогон. Добавить в образцы:")
        print(f"   python tools/split_templates.py {source} --apply")
        return 0

    cfg = Config.load()
    setup_logging(cfg)
    store = TemplateStore(cfg.resolve_path("storage.templates_dir"))
    kind = "proceedings" if "--proceedings" in sys.argv else "court"

    added = 0
    for index, chunk in enumerate(good, 1):
        # Имя — по виду документа, а не по шапке суда: шапка у всех
        # одинаковая, а генератору потом выбирать между определением,
        # решением и приговором. Вид стоит заглавными строкой ниже.
        lines_of = [x.strip() for x in chunk.split("\n") if x.strip()]
        kind_line = next(
            (x for x in lines_of[1:6]
             if x.isupper() and 4 < len(x) < 45 and "СУД" not in x),
            lines_of[0] if lines_of else "")
        detail = next((x for x in lines_of[1:8]
                       if not x.isupper() and 10 < len(x) < 70), "")
        name = re.sub(r"[^\w\s.-]", "",
                      f"{kind_line} {detail}")[:60].strip() or f"образец {index}"
        try:
            store.add(kind, chunk, f"{index:02d} {name}")
            added += 1
        except ValueError as exc:
            print(f"  {index:2}. не добавлен: {exc}")

    print(f"\nДобавлено в «{KINDS[kind]}»: {added}")
    print(f"Всего образцов: {store.counts()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
