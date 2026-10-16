"""
Импорт правил из буфера обмена (ТЗ 5.3, хоткей import_laws).

Сценарий ровно такой, как в ТЗ: пользователь сам открывает страницу
форума, жмёт Ctrl+A и Ctrl+C, затем хоткей импорта. Никакой
автоматизации браузера и обхода анти-бот защиты — приложение работает
только с тем, что пользователь скопировал руками.
"""
from __future__ import annotations

import logging

from core.clipboard import ClipboardError, get_text
from core.legal.db import ImportResult, LegalDB
from core.legal.parser import ForumParser

log = logging.getLogger(__name__)

# Ниже этого объёма текст почти наверняка не страница правил, а случайно
# скопированная строка. Импортировать её — значит замусорить базу.
MIN_TEXT_LENGTH = 500

# Какую долю похожих на пункт строк парсер обязан разобрать, чтобы
# импорту можно было верить. Абсолютный порог тут не годится: страница
# «Правила проведения допросов с применением насильственных действий»
# состоит всего из двух пунктов, и это нормальная страница, а не сбой.
# Важно не «сколько пунктов нашли», а «много ли потеряли».
MIN_PARSED_RATIO = 0.5


class ImportError_(RuntimeError):
    """Импортировать нечего или текст не похож на правила."""


class LawImporter:
    def __init__(self, db: LegalDB, parser: ForumParser):
        self.db = db
        self.parser = parser

    def import_from_clipboard(self, title: str = "") -> ImportResult:
        return self.import_text(self._clipboard(), title=title)

    def import_reference_from_clipboard(self, title: str = "") -> ImportResult:
        return self.import_reference(self._clipboard(), title=title)

    @staticmethod
    def _clipboard() -> str:
        try:
            return get_text()
        except ClipboardError as exc:
            raise ImportError_(str(exc)) from exc

    def import_reference(self, raw: str, title: str = "") -> ImportResult:
        """Импорт справочной страницы: услуги, прайсы, инструкции.

        Нумерации тут не ждём и не требуем — иначе прайс-лист EMS
        или перечень услуг агентства в базу не попадут никогда.
        """
        if len(raw.strip()) < MIN_TEXT_LENGTH:
            raise ImportError_(
                f"В буфере всего {len(raw.strip())} символов — "
                "скопируйте страницу целиком: Ctrl+A, Ctrl+C.")
        parsed = self.parser.parse_reference(raw, title=title)
        if not parsed.rules:
            raise ImportError_(
                "На странице не нашлось ни одной содержательной строки. "
                f"Заголовок: «{parsed.title}».")
        result = self.db.import_document(parsed, raw_text=raw)
        log.info("Справочный импорт: %s", result.summary())
        return result

    def import_text(self, raw: str, title: str = "") -> ImportResult:
        if len(raw.strip()) < MIN_TEXT_LENGTH:
            raise ImportError_(
                f"В буфере всего {len(raw.strip())} символов — это не похоже "
                "на страницу правил. Скопируйте страницу целиком: Ctrl+A, Ctrl+C.")

        parsed = self.parser.parse(raw, title=title)
        # Кандидатов считает сам парсер и только по первому посту.
        # Считать их по всему тексту нельзя: в темах-реестрах вроде
        # «Пропуска на посещения закрытых территорий» ниже идут сотни
        # постов с выданными пропусками, и диагностика врала про
        # «0 из 60 потеряно», хотя разбирался только первый пост.
        candidates = parsed.candidate_lines

        if not parsed.rules:
            raise ImportError_(self._explain_failure(parsed, candidates))
        if candidates and len(parsed.rules) < candidates * MIN_PARSED_RATIO:
            raise ImportError_(self._explain_failure(parsed, candidates))

        # Исходник сохраняем вместе с разобранным: когда парсер научится
        # чему-то новому, документы можно будет переразобрать на месте,
        # не копируя каждую страницу с форума заново.
        result = self.db.import_document(parsed, raw_text=raw)
        log.info("Импорт из буфера: %s", result.summary())
        return result

    def _explain_failure(self, parsed, candidates: int) -> str:
        """Объясняет, почему не вышло — по-разному в двух разных случаях.

        Страница без единой нумерованной строки — это объявление или
        новость, а не правила: чинить тут нечего, и советовать правку
        конфига было бы враньём. Другое дело, когда нумерованные строки
        есть, а разобрались не все: вот тогда шаблон действительно не
        совпал с форматом.
        """
        if candidates == 0:
            return ("На этой странице нет ни одного пронумерованного "
                    f"пункта — похоже, это объявление или новость, "
                    f"а не правила. Заголовок: «{parsed.title}». "
                    "Импортировать нечего, ничего чинить не нужно.")

        return (f"Разобрано пунктов: {len(parsed.rules)} из "
                f"{candidates} похожих на пункт строк — потеряно слишком "
                "много. Формат этой страницы отличается от уже освоенных: "
                f"покажите её мне. Заголовок: «{parsed.title}».")
