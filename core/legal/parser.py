"""
Разбор текста правил, скопированного с форума Eclipse (ТЗ 5.3).

Формат выведен из реального Ctrl+A/Ctrl+C страницы «Общие правила
проекта», а не придуман. Ключевые наблюдения:

* Раздел — это строка, за которой идёт строка «gta5_divider.png»
  (альт-текст картинки-разделителя). Признак надёжный: в проверенном
  документе так размечены все 7 разделов и ничего лишнего.
* Пункт — строка вида «1.1», «6.26», «1.7.1» плюс текст.
* Санкция идёт после символа «|», но НЕ каждый «|» — санкция: в примерах
  RP-процесса встречается «/try взломал замок | Неудачно.».
* «Примечание:» и «Исключение:» — отдельные строки, относящиеся
  к предыдущему пункту.
* Копирование склеивает заголовок спойлера со следующим пунктом:
  «...зона запрещенная для агитации6.26 Похищения...».
* В конце страницы идёт второй пост «#2» — журнал изменений, дословно
  повторяющий часть пунктов. Его нужно отрезать, иначе пункты задвоятся.

Все шаблоны вынесены в config/config.yaml, секция legal_parser: формат
других разделов форума (правила фракций, зелёных зон) уточняется по мере
их импорта, и править их нужно без изменения кода.
"""
from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# Картинка-разделитель, которой форум отбивает заголовки разделов.
# Имя файла у неё не одно: в общих правилах это «gta5_divider.png»,
# в правилах допросов — «K61Vjxh.png». Поэтому ловим любую строку,
# состоящую только из имени файла картинки.
# Имя файла бывает и с пробелами: «Desktop Screenshot 2025.10.21 -
# 00.34.36.62.png», «ChatGPT_Image_1_._2026_._14_36_16 (1).png».
# Без учёта пробелов такие строки становились названиями разделов.
DEFAULT_SECTION_MARKER = r"^[^\n]{1,120}\.(?:png|gif|jpe?g|webp)$"

# Номер пункта в начале строки. Два вида нумерации, оба встречаются:
#   иерархическая — «1.1», «6.26», «1.7.1» (общие правила, зелёные зоны);
#   плоская       — «1.», «2.», «3.»       («Война за граффити»).
# Для плоской точка после номера обязательна, иначе под пункт попадут
# строки вроде «30 Июн 2026» и «2 Авг 2026» из блока автора.
DEFAULT_RULE_PATTERN = r"^(\d+(?:\.\d+)+|\d+\.)[.\s]\s*(.+)$"

# Заголовок спойлера, приклеившийся к номеру следующего пункта.
# Требуем букву перед номером и заглавную букву после — иначе разорвём
# безобидные «п.1.1» и «RageMP 4.5».
DEFAULT_GLUE_PATTERN = r"(?<=[а-яёa-z])(\d+(?:\.\d+)+)\s+(?=[А-ЯЁA-Z])"

# Начало нового поста в теме: «#1», «#2». Всё, что после первого поста, —
# журнал изменений, а не текст правил.
DEFAULT_POST_PATTERN = r"^#(\d+)\s*$"

# Заголовок главы: «Глава I Общие сведения.», «Глава 2. Форма».
# Встречается в уставах и положениях госструктур. Там же вскрывается
# третий вид разметки: внутри главы пункты идут БЕЗ номеров, обычными
# предложениями, и без этого шаблона такая страница не даёт ни одного
# пункта вообще.
DEFAULT_CHAPTER_PATTERN = r"(?i)^\s*глава\s+([IVXLCDM]+|\d+)[.\s]+(.*)$"

# «Статья 2.1 Сотрудник правоохранительных органов обязан...» — так
# устроены кодексы и законы штата: номер статьи стоит не отдельной
# строкой, а в начале абзаца внутри главы. Без его извлечения пункт
# хранится под позиционным номером вроде «I.3», и ни сослаться на
# «ст. 2.1», ни найти её по номеру нельзя.
# Звёздочки после номера («Статья 17.1***») помечают тяжесть статьи и
# частью текста не являются — иначе пункт начинается с «*** (РЕГИОНАЛЬНО...».
DEFAULT_ARTICLE_PATTERN = r"^Стать[яи]\s+(\d+(?:\.\d+)*)[.*\s]*(.*)$"

# Подпункты статьи: «а) Если сотрудник стал очевидцем...». Отдельными
# пунктами их делать нельзя — в отрыве от своей статьи они бессмысленны.
DEFAULT_SUBITEM_PATTERN = r"^(?:[а-яёa-z]\)|[-—•])\s*"

# Санкция в конце строки через тире: «...в виде денег. - от 3 до 6 лет
# лишения свободы. ((от 30 до 60 мин))». Именно так оформлены сроки
# в Уголовном кодексе, и без разбора они тонут внутри текста статьи.
# Служебные вставки форума, приклеенные к тексту пункта в середине
# строки. Построчная чистка их не берёт: «Спойлер: зона запрещенная для
# агитации» стоит в одной строке с нормой, а «Последнее редактирование»
# дописывается к последнему пункту темы. Всё от такой вставки и до
# конца строки отрезается.
DEFAULT_INLINE_CHROME = (
    r"\s*(?:Спойлер:|Нажмите, чтобы раскрыть|Последнее редактирование|"
    r"Реакции:|Добавить закладку|Напишите свой ответ).*$")

DEFAULT_INLINE_SANCTION = (
    r"\s[-—]\s+((?:от\s+\d|до\s+\d|штраф|лишени|арест|деморган|"
    r"предупрежд).*)$")

DEFAULT_NOTE_PREFIXES = ("Примечание:", "Исключение:", "Пример:")

# Служебные строки форума. Между последним пунктом и следующим постом
# лежит подвал поста — кнопки реакций и блок автора. Без этого списка
# он приклеивается к последнему пункту документа.
DEFAULT_CHROME_PATTERNS = (
    r"^Like",
    r"^Реакции:",
    r"^Добавить закладку$",
    r"^Поделиться:$",
    r"^Пожаловаться$",
    r"^Отслеживать$",
    r"^Автор темы",
    r"^В этой теме нельзя размещать",
    r"^\[(?:Пункт|Добавлен|Изменен|Удалён|Удален)",
    # «РАЗДЕЛ II — СЛУЖБА И СТАТУС СОТРУДНИКА» в уставе FIB: это
    # надзаголовок над главами, а не пункт. Без этого он попадал
    # в текст как содержимое ненумерованной главы.
    r"(?i)^раздел\s+[IVXLCDM\d]+\b",
)

# Слова, по которым хвост после «|» опознаётся как санкция, а не как
# часть примера. Список расширяется в конфиге по мере импорта.
DEFAULT_SANCTION_WORDS = (
    "блокировк", "перманент", "черный список", "чёрный список",
    "обнулени", "warn", "варн", "деморган", "принудительн",
    "запрет", "смена", "предупреждени", "изъяти", "штраф",
    "выговор", "увольнени", "разжаловани", "арест",
    # Фракционные санкции: штрафные баллы фракции и лидеру, мут в чате.
    # Без них 40 пунктов из 461 остались с «|» внутри текста.
    "балл", "мут ", "мут.", "замут",
    "приравнивается", "наказание", "откат",
    # Основная санкция Уголовного кодекса. Без неё срок оставался
    # внутри текста статьи, а поле санкции пустовало: из 142 статей УК
    # санкция была распознана у четырёх.
    "лишени", "срок ", "лет ", "год ", "года ", "изъятие", "конфискаци",
    "обязательн", "исправительн", "принудительн",
)


def _clean_text(raw: str) -> str:
    """Убирает BOM, неразрывные пробелы и невидимые символы форума."""
    text = raw.replace("﻿", "")
    text = unicodedata.normalize("NFKC", text)
    # Форум щедро сыплет ​ (zero-width space) в заголовки.
    text = re.sub(r"[​‌‍⁠]", "", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def strip_bbcode(text: str) -> str:
    """Срезает BBCode, если он вдруг попал в буфер (копирование из цитаты)."""
    text = re.sub(r"\[/?(?:b|i|u|s|color|size|font|center|left|right|"
                  r"quote|spoiler|url|img|list|\*)(?:=[^\]]*)?\]",
                  "", text, flags=re.IGNORECASE)
    return text


def _looks_like_rule_line(line: str) -> bool:
    """Строка похожа на пункт, пусть и в незнакомом формате.

    Служит только диагностике: отличает «на странице нет правил» от
    «правила есть, но шаблон не подошёл». Даты блока автора под это
    подходить не должны.
    """
    stripped = line.strip()
    if len(stripped) < 30 or not stripped[:1].isdigit():
        return False
    return not re.match(r"^\d{1,2}\s+[А-Яа-я]{3,8}\s+\d{4}$", stripped)


@dataclass
class Rule:
    number: str                       # «1.7.1»
    section_number: str               # «1»
    section_title: str                # «Общее положение»
    text: str                         # текст пункта без санкции
    sanction: str = ""                # то, что после «|»
    notes: list[str] = field(default_factory=list)

    @property
    def title(self) -> str:
        """Короткий заголовок для показа в оверлее."""
        first = self.text.split(". ")[0].strip()
        return first if len(first) <= 90 else first[:87] + "..."

    @property
    def full_text(self) -> str:
        parts = [self.text]
        if self.sanction:
            parts.append(f"Санкция: {self.sanction}")
        parts.extend(self.notes)
        return "\n".join(parts)

    @property
    def hash(self) -> str:
        """Хэш содержимого — по нему ловим изменение пункта при реимпорте."""
        payload = f"{self.number}\x00{self.full_text}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# Строка шапки темы: « Автор темыappolo  Дата начала24 Июл 2026».
# Даёт автора и дату создания — единственное, чем можно развести две
# разные темы форума с одинаковым названием.
DEFAULT_THREAD_META_PATTERN = r"Автор темы(.*?)\s*Дата начала\s*(.+?)\s*$"


@dataclass
class ParsedDocument:
    title: str
    rules: list[Rule] = field(default_factory=list)
    sections: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    author: str = ""
    started_at: str = ""
    # "rules" — нумерованные правила, "reference" — справочная страница
    # (услуги, прайсы, инструкции). У справочной нет нумерации, и
    # требовать её от прайс-листа бессмысленно.
    kind: str = "rules"
    # Строк, похожих на пункт, в первом посте. Считать их по всему
    # тексту нельзя: в темах-реестрах ниже идут сотни постов, и
    # диагностика начинает врать про «потеряно слишком много».
    candidate_lines: int = 0

    @property
    def hash(self) -> str:
        joined = "\x00".join(rule.hash for rule in self.rules)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()


class ForumParser:
    def __init__(self, *, section_marker: str = DEFAULT_SECTION_MARKER,
                 rule_pattern: str = DEFAULT_RULE_PATTERN,
                 glue_pattern: str = DEFAULT_GLUE_PATTERN,
                 post_pattern: str = DEFAULT_POST_PATTERN,
                 note_prefixes: tuple[str, ...] = DEFAULT_NOTE_PREFIXES,
                 chrome_patterns: tuple[str, ...] = DEFAULT_CHROME_PATTERNS,
                 chapter_pattern: str = DEFAULT_CHAPTER_PATTERN,
                 article_pattern: str = DEFAULT_ARTICLE_PATTERN,
                 subitem_pattern: str = DEFAULT_SUBITEM_PATTERN,
                 inline_sanction: str = DEFAULT_INLINE_SANCTION,
                 inline_chrome: str = DEFAULT_INLINE_CHROME,
                 sanction_words: tuple[str, ...] = DEFAULT_SANCTION_WORDS,
                 min_rule_length: int = 20,
                 max_section_title_length: int = 80,
                 strip_bbcode_enabled: bool = True):
        self.section_re = re.compile(section_marker)
        self.rule_re = re.compile(rule_pattern)
        self.glue_re = re.compile(glue_pattern)
        self.post_re = re.compile(post_pattern)
        self.chrome_res = [re.compile(p) for p in chrome_patterns]
        self.thread_meta_re = re.compile(DEFAULT_THREAD_META_PATTERN)
        self.chapter_re = re.compile(chapter_pattern)
        self.article_re = re.compile(article_pattern)
        self.subitem_re = re.compile(subitem_pattern)
        self.inline_sanction_re = re.compile(inline_sanction, re.IGNORECASE)
        self.inline_chrome_re = re.compile(inline_chrome, re.IGNORECASE)
        self.note_prefixes = note_prefixes
        self.sanction_words = tuple(w.lower() for w in sanction_words)
        self.min_rule_length = min_rule_length
        self.max_section_title_length = max_section_title_length
        self.strip_bbcode_enabled = strip_bbcode_enabled

    @classmethod
    def from_config(cls, cfg) -> "ForumParser":
        section = cfg.get("legal_parser", {}) or {}
        return cls(
            section_marker=section.get("section_marker", DEFAULT_SECTION_MARKER),
            rule_pattern=section.get("rule_pattern", DEFAULT_RULE_PATTERN),
            glue_pattern=section.get("glue_pattern", DEFAULT_GLUE_PATTERN),
            post_pattern=section.get("post_pattern", DEFAULT_POST_PATTERN),
            note_prefixes=tuple(section.get("note_prefixes",
                                            DEFAULT_NOTE_PREFIXES)),
            chrome_patterns=tuple(section.get("chrome_patterns",
                                              DEFAULT_CHROME_PATTERNS)),
            chapter_pattern=section.get("chapter_pattern",
                                        DEFAULT_CHAPTER_PATTERN),
            article_pattern=section.get("article_pattern",
                                        DEFAULT_ARTICLE_PATTERN),
            subitem_pattern=section.get("subitem_pattern",
                                        DEFAULT_SUBITEM_PATTERN),
            inline_sanction=section.get("inline_sanction",
                                        DEFAULT_INLINE_SANCTION),
            inline_chrome=section.get("inline_chrome",
                                      DEFAULT_INLINE_CHROME),
            sanction_words=tuple(section.get("sanction_words",
                                             DEFAULT_SANCTION_WORDS)),
            min_rule_length=section.get("min_rule_length", 20),
            max_section_title_length=section.get(
                "max_section_title_length", 80),
            strip_bbcode_enabled=section.get("strip_bbcode", True),
        )

    # ------------------------------------------------------------- подготовка
    def _select_first_post(self, lines: list[str],
                           warnings: list[str]) -> list[str]:
        """Собирает текст документа из постов темы.

        Длинный устав не помещается в один пост: «Устав Federal
        Investigation Bureau» разложен на три поста, «Устав EMS» — на
        восемь, и продолжение пишет тот же автор темы. Обрезать всё
        после первого поста — значит потерять половину документа:
        у FIB так терялось 97 пунктов из 216, включая целую главу про
        отдел внутренних расследований.

        Отличить продолжение от журнала изменений можно по нумерации:
        продолжение её продолжает (после 10.x идёт 11.1), а журнал
        изменений повторяет уже встречавшиеся номера. На повторе и
        останавливаемся.
        """
        starts: list[int] = []
        for index, line in enumerate(lines):
            if self.post_re.match(line.strip()):
                starts.append(index)

        if not starts:
            warnings.append(
                "Границы постов (#1, #2) не найдены — шапка и подвал форума "
                "могли попасть в текст.")
            return lines

        bounds = [(start + 1, starts[i + 1] if i + 1 < len(starts) else len(lines))
                  for i, start in enumerate(starts)]

        collected = list(lines[bounds[0][0]:bounds[0][1]])
        seen = self._numbers_in(collected)

        if not seen:
            # В первом посте нумерации нет — продолжать нечего. Иначе
            # в документ заезжает реестр: в теме «Пропуска на посещения
            # закрытых территорий» каждый выданный пропуск оформлен
            # пронумерованной формой 1..5, и она притворяется главой.
            return collected

        for index, (begin, end) in enumerate(bounds[1:], start=2):
            chunk = lines[begin:end]
            numbers = self._numbers_in(chunk)
            if not numbers:
                continue      # пост без пунктов: реакции, благодарности

            repeated = numbers & seen
            if repeated:
                log.info("Пост #%d повторяет пункты %s — принят за журнал "
                         "изменений, дальше не читаем",
                         index, ", ".join(sorted(repeated)[:3]))
                break

            # Продолжение обязано продолжать нумерацию: после главы 10
            # идёт 11.1, а не новая единица. Пост, начинающий счёт заново,
            # документу не принадлежит.
            if self._max_section(numbers) < self._max_section(seen):
                log.info("Пост #%d начинает нумерацию заново — не часть "
                         "документа, дальше не читаем", index)
                break

            collected.extend(chunk)
            seen |= numbers
            log.info("Пост #%d — продолжение документа, добавлено пунктов: %d",
                     index, len(numbers))

        return collected

    @staticmethod
    def _max_section(numbers: set[str]) -> int:
        """Наибольший номер раздела среди номеров пунктов."""
        sections = [int(n.split(".")[0]) for n in numbers
                    if n.split(".")[0].isdigit()]
        return max(sections) if sections else 0

    def _numbers_in(self, lines: list[str]) -> set[str]:
        """Номера пунктов и статей, встречающиеся в куске текста.

        Статьи учитывать обязательно. В кодексах номер записан словом —
        «Статья 16.1», а не «16.1», — и без этого второй пост Уголовного
        кодекса выглядел пустым и отбрасывался. Вместе с ним терялась
        вся Особенная часть: 92 статьи, то есть сами составы
        преступлений, ради которых кодекс и нужен.
        """
        found: set[str] = set()
        for line in lines:
            stripped = line.strip()
            article = self.article_re.match(stripped)
            if article and self._valid_number(article.group(1)):
                found.add(article.group(1))
                continue
            for piece in self._unglue(stripped):
                match = self.rule_re.match(piece.strip())
                if match:
                    number = match.group(1).rstrip(".")
                    if self._valid_number(number):
                        found.add(number)
        return found

    # Заголовок статьи, оказавшийся в середине абзаца. При копировании
    # с форума пустые строки между статьями иногда теряются, и целый
    # раздел приезжает одним куском: «...Статья 4 Сотрудник EMS...
    # Статья 5.1 Правило Миранды: Вы имеете право хранить молчание...».
    # Парсер ищет «Статья N» только в начале абзаца, поэтому весь этот
    # кусок приклеивался к предыдущей статье — в Процессуальном кодексе
    # так слиплись девять тысяч символов, и правило Миранды перестало
    # находиться вовсе.
    #
    # Требуется именно именительный падеж: ссылки внутри текста звучат
    # как «в соответствии со Статьей 1.1» и «пунктом б) Статьи 2.2»,
    # и резать по ним нельзя — это не заголовки, а перекрёстные ссылки.
    INLINE_ARTICLE = re.compile(r"\s+(?=Статья\s+\d+(?:\.\d+)*\s)")

    def _unglue(self, line: str) -> list[str]:
        """Разрывает строку, где заголовок слип с текстом предыдущего."""
        repaired = self.glue_re.sub(lambda m: "\n" + m.group(1) + " ", line)
        repaired = self.INLINE_ARTICLE.sub("\n", repaired)
        return repaired.split("\n")

    def _is_sanction(self, tail: str) -> bool:
        lowered = tail.lower()
        return any(word in lowered for word in self.sanction_words)

    def strip_inline_chrome(self, text: str) -> str:
        """Срезает служебную вставку форума и всё, что за ней.

        «Спойлер: зона запрещенная для агитации» приклеивается к норме
        в той же строке, а «Последнее редактирование» дописывается к
        последнему пункту темы. Построчная чистка их не видит, и обе
        попадают в текст статьи — а оттуда в ответ пользователю.
        """
        return self.inline_chrome_re.sub("", text).strip()

    def _split_sanction(self, text: str) -> tuple[str, str]:
        """Отделяет санкцию от текста пункта.

        Режем по последнему «|», но только если хвост похож на санкцию:
        в примерах RP-процесса «|» разделяет совсем другое.
        """
        text = self.strip_inline_chrome(text)
        # Ведущая черта — не санкция, а разделитель после номера:
        # «1.1 | Назначение Устава. Настоящий Устав...» в уставе FIB.
        # Без этого она оставалась в тексте у 109 пунктов из 119.
        text = text.lstrip().lstrip("|").lstrip()

        if "|" not in text:
            # В кодексах санкция отделена не чертой, а тире, и стоит
            # в конце той же строки: «...взятки в виде денег. - от 3 до
            # 6 лет лишения свободы. ((от 30 до 60 мин))». Без этого
            # срок оставался внутри текста, а поле санкции пустовало —
            # и в ответе не было главного, за что статья и сколько дают.
            match = self.inline_sanction_re.search(text)
            if match and self._is_sanction(match.group(1)):
                return text[:match.start()].strip(), match.group(1).strip()
            return text.strip(), ""
        head, _, tail = text.rpartition("|")
        tail = tail.strip()
        if not self._is_sanction(tail):
            return text.strip(), ""
        return head.strip(), tail

    # ---------------------------------------------------------------- разбор
    def parse(self, raw: str, *, title: str = "") -> ParsedDocument:
        warnings: list[str] = []
        text = _clean_text(raw)
        if self.strip_bbcode_enabled:
            text = strip_bbcode(text)

        all_lines = text.split("\n")
        document_title = title or self._guess_title(all_lines)
        # Автора и дату темы берём ДО отсечения — они в шапке форума,
        # то есть в том куске, который _select_first_post выбрасывает.
        author, started_at = self._thread_meta(all_lines)
        lines = self._select_first_post(all_lines, warnings)

        # Разделы помечены строкой-картинкой ПОСЛЕ заголовка, поэтому
        # сначала находим их позиции, а потом идём по тексту сверху вниз.
        section_at: dict[int, str] = {}
        for index, line in enumerate(lines):
            if self.section_re.match(line.strip()):
                heading = lines[index - 1].strip() if index else ""
                if self._is_section_heading(heading):
                    section_at[index] = heading

        document = ParsedDocument(title=document_title, warnings=warnings,
                                  author=author, started_at=started_at)
        section_number = "1"
        section_title = ""
        section_index = 1
        seen_numbers: dict[str, int] = {}
        in_chapter = False
        chapter_had_rules = False
        chapter_lines: list[str] = []
        current: Rule | None = None
        pending: list[str] = []

        def close_current() -> None:
            nonlocal current, pending
            if current is None:
                return
            if pending:
                extra = " ".join(p.strip() for p in pending if p.strip())
                if extra:
                    body, sanction = self._split_sanction(extra)
                    if sanction and not current.sanction:
                        # Санкция может стоять не в строке с номером:
                        # у пункта 2.7 она отделена от него дюжиной строк.
                        current.sanction = sanction
                    if body:
                        current.text = f"{current.text} {body}".strip()
            if len(current.full_text) >= self.min_rule_length:
                document.rules.append(current)
            else:
                warnings.append(
                    f"Пункт {current.number} короче {self.min_rule_length} "
                    "символов, пропущен как мусор.")
            current, pending = None, []

        def flush_chapter() -> None:
            """Превращает абзацы ненумерованной главы в пункты.

            «Пропуска на посещения закрытых территорий» размечены главами
            «Глава I», «Глава II», а требования внутри идут обычными
            предложениями без номеров. Раньше такая страница давала ноль
            пунктов и содержимое терялось целиком. Номер здесь не выдуман
            из воздуха: это позиция абзаца внутри своей главы, и в цитате
            она честно выглядит как «Глава I, п. 3».
            """
            nonlocal chapter_lines, chapter_had_rules
            if chapter_had_rules or not chapter_lines:
                chapter_lines = []
                return
            position = 0
            last: Rule | None = None
            # Глава со статьями устроена иначе, чем глава из голых
            # абзацев: в ней каждый абзац без номера — продолжение
            # предыдущей статьи. В Дорожном кодексе статья 42 звучит
            # «На автомагистралях запрещается:», а сам перечень запретов
            # идёт следующими строками без всяких маркеров. Нумеровать их
            # по позиции — значит оторвать запреты от статьи, а штраф
            # приклеить к последнему из них.
            has_articles = any(self.article_re.match(p) for p in chapter_lines)

            for paragraph in chapter_lines:
                # Подпункт «а) ...» дописываем к своей статье: сам по
                # себе он бессмыслен, а как отдельный пункт ещё и рвёт
                # статью на куски при поиске.
                if last is not None and self.subitem_re.match(paragraph):
                    tail = self.subitem_re.sub("", paragraph).strip()
                    if not last.sanction and self._is_sanction(tail):
                        # «- штраф до 800$.» отдельной строкой: это
                        # санкция статьи, а не ещё один её пункт.
                        last.sanction = tail
                    else:
                        last.text = f"{last.text} {paragraph}".strip()
                    continue
                if paragraph.startswith(self.note_prefixes):
                    if last is not None:
                        last.notes.append(paragraph)
                        continue

                article = self.article_re.match(paragraph)
                if article:
                    # Настоящий номер статьи из текста — он и пойдёт
                    # в цитату: «Уголовный Кодекс, ст. 1.3».
                    # Текст в той же строке не обязателен: статьи о
                    # дорожных знаках выглядят как «Статья 58», дальше
                    # картинка знака, и только потом описание. Требовать
                    # текст сразу — значит потерять семь таких статей,
                    # приклеив их описания к предыдущей.
                    raw_number, paragraph = article.group(1), article.group(2).strip()
                elif len(paragraph) < self.min_rule_length:
                    continue
                elif has_articles:
                    if last is not None:
                        last.text = f"{last.text} {paragraph}".strip()
                    continue
                else:
                    position += 1
                    raw_number = f"{section_number}.{position}"

                body, sanction = self._split_sanction(paragraph)
                # Через ту же проверку на повтор, что и нумерованные
                # пункты: иначе абзац главы «Заключительные положения»
                # получал номер 3.1 и молча затирал настоящий пункт 3.1
                # из главы «Компетенция подразделений».
                number = self._unique_number(raw_number, seen_numbers,
                                             document, section_number)
                last = Rule(
                    number=number,
                    section_number=section_number,
                    section_title=section_title,
                    text=body,
                    sanction=sanction,
                )
                document.rules.append(last)

            # Статья, оставшаяся без текста (например, за картинкой знака
            # не оказалось описания), в базе бесполезна.
            document.rules = [r for r in document.rules
                              if r.text.strip() or r.sanction]
            chapter_lines = []

        for index, source_line in enumerate(lines):
            if index in section_at:
                close_current()
                if section_title:
                    # Первый заголовок не увеличивает счётчик: он и есть
                    # раздел №1.
                    section_index += 1
                section_title = section_at[index]
                continue
            if self.section_re.match(source_line.strip()):
                continue

            for line in self._unglue(source_line):
                stripped = line.strip()
                if not stripped:
                    continue

                chapter = self.chapter_re.match(stripped)
                if chapter:
                    close_current()
                    flush_chapter()
                    section_number = chapter.group(1)
                    section_title = (chapter.group(2).strip().rstrip(".")
                                     or f"Глава {section_number}")
                    document.sections[section_number] = section_title
                    in_chapter = True
                    chapter_had_rules = False
                    chapter_lines = []
                    continue

                # «Статья 5.1 Правило Миранды: ...» — такое же начало
                # пункта, как «5.1 ...», просто со словом впереди. Без
                # этой проверки статьи узнавались только внутри глав без
                # нумерации, а стоило в главе встретиться хоть одному
                # пункту вида «2.», как все последующие статьи уходили
                # в хвост предыдущего. В Процессуальном кодексе так
                # слиплись девять тысяч символов: тринадцать статей,
                # включая правило Миранды, переставали находиться.
                article = self.article_re.match(stripped)
                if article and self._valid_number(article.group(1)):
                    close_current()
                    number = article.group(1)
                    body, sanction = self._split_sanction(
                        article.group(2).strip())
                    section_number = (number.split(".")[0] if "." in number
                                      else str(section_index))
                    if section_title:
                        document.sections[section_number] = section_title
                    current = Rule(
                        number=self._unique_number(number, seen_numbers,
                                                   document),
                        section_number=section_number,
                        section_title=section_title,
                        text=body,
                        sanction=sanction,
                    )
                    continue

                match = self.rule_re.match(stripped)
                if match and self._valid_number(
                        match.group(1).rstrip(".")):
                    close_current()
                    number, body = match.group(1), match.group(2)
                    number = number.rstrip(".")   # «1.» -> «1»

                    if self._is_numbered_heading(number, body):
                        # «1. Общие положения» — это заголовок раздела,
                        # а не пункт: дальше пойдут 1.1, 1.2, 1.3.
                        # Раньше такая строка отбрасывалась как слишком
                        # короткая, и все разделы устава Weazel News
                        # остались безымянными.
                        close_current()
                        flush_chapter()
                        section_number = number
                        section_title = body.strip().rstrip(".")
                        document.sections[section_number] = section_title
                        in_chapter = False
                        chapter_lines = []
                        continue

                    body, sanction = self._split_sanction(body)
                    # В главе нашлась нормальная нумерация — значит
                    # достраивать пункты из абзацев не нужно.
                    chapter_had_rules = True

                    if "." in number:
                        # Иерархическая нумерация: ведущая цифра и есть
                        # номер раздела — проверено на реальных документах.
                        section_number = number.split(".")[0]
                    else:
                        # Плоская нумерация: номер пункта про раздел ничего
                        # не говорит, иначе каждый пункт стал бы отдельным
                        # разделом. Считаем разделы по порядку появления.
                        section_number = str(section_index)

                    if section_title:
                        document.sections[section_number] = section_title

                    number = self._unique_number(number, seen_numbers, document)

                    current = Rule(
                        number=number,
                        section_number=section_number,
                        section_title=section_title,
                        text=body,
                        sanction=sanction,
                    )
                    continue

                if self._is_post_footer(stripped):
                    # Подвал поста: «Последнее редактирование», реакции,
                    # блок автора. В ненумерованной главе их некому
                    # отсечь, и они становились статьями III.7-III.9.
                    close_current()
                    flush_chapter()
                    in_chapter = False
                    continue

                if self._is_chrome(stripped):
                    # Прочие служебные строки форума.
                    close_current()
                    continue

                if current is None:
                    if in_chapter and not self.section_re.match(stripped):
                        # Глава без нумерации: копим её абзацы, чтобы не
                        # потерять содержимое целиком.
                        chapter_lines.append(stripped)
                    continue

                if stripped.startswith(self.note_prefixes):
                    current.notes.append(stripped)
                else:
                    pending.append(stripped)

        close_current()
        flush_chapter()
        # Кандидатов считаем только по первому посту — именно его мы
        # и разбираем.
        document.candidate_lines = sum(
            1 for line in lines if _looks_like_rule_line(line))
        self._check_gaps(document)
        return document

    def _is_chrome(self, line: str) -> bool:
        return any(pattern.match(line) for pattern in self.chrome_res)

    @staticmethod
    def _is_post_footer(line: str) -> bool:
        """Начало подвала поста: дальше содержимого документа уже нет."""
        return bool(re.match(
            r"^(?:Like|Реакции:|Поделиться:|Напишите свой ответ|"
            r"English \(US\)|В этой теме нельзя размещать|Вложения|"
            r"Последнее редактирование)",
            line))

    def _thread_meta(self, lines: list[str]) -> tuple[str, str]:
        """Автор темы и дата её создания из шапки форума.

        Нужны не для красоты: названия тем на форуме не уникальны —
        «Правила проведения допросов» встречается дважды, и это две
        разные темы с разным содержимым. Дата создания разводит их.
        """
        for line in lines[:30]:
            match = self.thread_meta_re.search(line)
            if match:
                return match.group(1).strip(), match.group(2).strip()
        return "", ""

    @staticmethod
    def _strip_forum_tag(title: str) -> str:
        """Срезает форумную метку темы, прилипшую к заголовку.

        Форум выводит её вплотную: «Важно Правила организации и
        проведения мероприятий». Меткой она быть перестаёт, а частью
        названия документа так и не становится.
        """
        for tag in ("Важно", "Закреплено", "Опрос", "Объявление",
                    "Архив", "Внимание"):
            if title.startswith(tag + " "):
                return title[len(tag) + 1:].strip()
        return title

    @staticmethod
    def _unique_number(number: str, seen: dict[str, int], document,
                       prefix: str = "") -> str:
        """Разводит повторяющиеся номера внутри одного документа.

        Номера не всегда уникальны: во «Внутреннем уставе SASPA» блоки
        про отделы SDO и MPS оба пронумерованы 20.1-20.3 — тема одна,
        нумерация переиспользована. Без суффикса второй блок затирает
        первый прямо при импорте.
        """
        if number not in seen:
            seen[number] = 1
            return number

        # В кодексах нумерация статей начинается заново в каждой главе:
        # «Статья 1» есть и в «Процессе задержания», и в «Процессе
        # ареста». Приписать «(2)» — значит потерять смысл, поэтому
        # сначала пробуем развести номером главы: «II.1».
        if prefix and not number.startswith(f"{prefix}."):
            qualified = f"{prefix}.{number}"
            if qualified not in seen:
                seen[qualified] = 1
                return qualified

        seen[number] += 1
        distinct = f"{number} ({seen[number]})"
        document.warnings.append(
            f"Пункт {number} встречается в документе не первый раз — "
            f"сохранён как «{distinct}».")
        return distinct

    @staticmethod
    def _valid_number(number: str) -> bool:
        """Отсеивает то, что похоже на номер пункта, но им не является.

        Главный нарушитель — дата: «09.04.2024 21:25» из реестра выданных
        пропусков подходит под шаблон номера и уводит разбор в сторону.
        У настоящего номера пункта каждая часть короткая, а четырёхзначный
        год выдаёт дату с головой.
        """
        parts = number.split(".")
        if len(parts) > 4:
            return False
        return all(part.isdigit() and len(part) <= 3 for part in parts)

    def _is_numbered_heading(self, number: str, body: str) -> bool:
        """«1. Общие положения» — заголовок раздела, а не пункт.

        Признаки: номер без подномера, короткий текст без завершающей
        точки. Настоящий пункт с плоской нумерацией — это предложение:
        «1. Запрещены любые виды IC союзов.» — и точка на конце его
        выдаёт.
        """
        if "." in number:
            return False
        text = body.strip()
        if not text or len(text) > self.max_section_title_length:
            return False
        if text.endswith((".", ";", ":", "!", "?")):
            return False
        # В заголовке не бывает санкции.
        return "|" not in text

    def _is_section_heading(self, line: str) -> bool:
        """Отличает заголовок раздела от преамбулы перед ним.

        Картинкой-разделителем форум отбивает не только заголовки: в
        «Правилах зеленых зон» второй такой картинкой закрыт вводный
        абзац. Без проверки его последняя строка («...обязательным
        требованием для игры на нашем сервере.») становится названием
        раздела.

        Заголовок — короткий, начинается с заглавной и не заканчивается
        точкой. Обычное предложение не проходит хотя бы по одному пункту.
        """
        if not line or len(line) > self.max_section_title_length:
            return False
        if self.rule_re.match(line) or self._is_chrome(line):
            return False
        if self.article_re.match(line):
            # «Статья 58» перед картинкой знака — это заголовок статьи,
            # а не раздела. В Дорожном кодексе знаки размечены именно
            # так: строка «Статья 58», следом картинка знака, и только
            # потом описание со штрафом. Без этой проверки семь статей
            # о знаках (58-64) уходили в названия разделов и пропадали
            # из базы вместе со штрафами.
            return False
        if self.section_re.match(line):
            # Две картинки подряд: вторая не заголовок первой.
            return False
        if line.endswith((".", ",", ";", ":", "!", "?")):
            return False
        first = line[0]
        return not (first.isalpha() and first.islower())

    def parse_reference(self, raw: str, *, title: str = "") -> ParsedDocument:
        """Разбирает справочную страницу — услуги, прайсы, инструкции.

        Такие страницы не пронумерованы: ни «1.1», ни «Глава I» там нет,
        обычный разбор отдаёт ноль пунктов и страница теряется. А нужна
        она не меньше правил: на вопрос «сколько стоит лечение в EMS»
        ответ лежит именно в прайсе.

        Здесь ничего не выдумывается: строки сохраняются как есть,
        а номер — это порядок строки внутри своего заголовка.
        """
        warnings: list[str] = []
        text = _clean_text(raw)
        if self.strip_bbcode_enabled:
            text = strip_bbcode(text)

        all_lines = text.split("\n")
        document_title = title or self._guess_title(all_lines)
        author, started_at = self._thread_meta(all_lines)
        lines = self._select_first_post(all_lines, warnings)

        document = ParsedDocument(
            title=document_title, warnings=warnings, author=author,
            started_at=started_at, kind="reference")

        section_number = "1"
        section_title = ""
        position = 0
        section_index = 1

        for source_line in lines:
            # Картинка нередко слипается с текстом: «разделитель(1).pngВ
            # данном разделе Вы можете подать заявление...». Отрываем.
            stripped = re.sub(
                r"(\.(?:png|gif|jpe?g|webp))(?=[А-ЯЁA-Z])", r"\1\n",
                source_line.strip()).split("\n")[-1].strip()
            if not stripped:
                continue
            if self._is_post_footer(stripped):
                # Дальше идут реакции, блок автора и подвал сайта.
                # В справочнике их некому отсечь — пунктов нет, и
                # «Напишите свой ответ...» становится содержимым.
                break
            if self.section_re.match(stripped) or self._is_chrome(stripped):
                continue

            chapter = self.chapter_re.match(stripped)
            if chapter:
                section_number = chapter.group(1)
                section_title = (chapter.group(2).strip().rstrip(".")
                                 or f"Глава {section_number}")
                document.sections[section_number] = section_title
                position = 0
                continue

            # Заголовком считаем только строку с двоеточием на конце:
            # «Ценовая политика:», «Услуги EMS LS:», «Медицинские карты:».
            # Общий определитель заголовков тут не годится — под него
            # подходит строка прайса «Осмотр — 3.000$», и цены пропадали
            # из документа вместе с ней.
            if (stripped.endswith(":")
                    and len(stripped) <= self.max_section_title_length):
                section_index += 1
                section_number = str(section_index)
                section_title = stripped.rstrip(":")
                document.sections[section_number] = section_title
                position = 0
                # Заголовок не выбрасываем: он и сам по себе осмысленный
                # кусок текста, а терять строки в справочнике нельзя.

            # Порог ниже, чем у правил: строка прайса «Лечение — 5.000$»
            # короткая, но это ровно то, ради чего страница и нужна.
            if len(stripped) < 5:
                continue

            position += 1
            document.rules.append(Rule(
                number=f"{section_number}.{position}",
                section_number=section_number,
                section_title=section_title,
                text=stripped,
            ))

        if not document.rules:
            warnings.append("На странице не нашлось ни одной содержательной "
                            "строки.")
        return document

    def _guess_title(self, lines: list[str]) -> str:
        """Заголовок темы.

        Форум выводит его строкой прямо перед строкой с автором и датой
        («Автор темыappolo  Дата начала24 Июл 2026»). Это устойчивый
        признак: до него идут навигация и ник текущего пользователя,
        которые заголовком не являются.
        """
        for index, line in enumerate(lines[:40]):
            if "Дата начала" in line and index:
                # Назад идём через пустые строки: между заголовком и
                # строкой автора их может быть сколько угодно, и брать
                # ровно предыдущую — значит поймать пустоту и откатиться
                # на ник пользователя из шапки форума.
                for back in range(index - 1, -1, -1):
                    candidate = lines[back].strip()
                    if len(candidate) > 3:
                        return self._strip_forum_tag(candidate)
                break

        for line in lines[:12]:
            stripped = line.strip()
            if (len(stripped) > 8 and not stripped.startswith("Форум")
                    and "Дата начала" not in stripped
                    and stripped not in ("Главная", "Играть", "Магазин")):
                return stripped
        return "Без названия"

    def _check_gaps(self, document: ParsedDocument) -> None:
        """Предупреждает о дырах в нумерации — признак недокопированного текста."""
        by_section: dict[str, list[int]] = {}
        for rule in document.rules:
            parts = rule.number.split(".")
            if len(parts) != 2:
                continue   # подпункты вида 1.7.1 в проверке не участвуют
            if not (parts[0].isdigit() and parts[1].isdigit()):
                # Повторный номер получил суффикс: «20.1 (2)». В проверке
                # пропусков он не участвует — это копия, а не новый пункт.
                continue
            by_section.setdefault(parts[0], []).append(int(parts[1]))

        for section, numbers in sorted(by_section.items()):
            numbers.sort()
            missing = [n for n in range(1, numbers[-1] + 1) if n not in numbers]
            if missing:
                document.warnings.append(
                    f"В разделе {section} пропущены пункты: "
                    + ", ".join(f"{section}.{n}" for n in missing))
