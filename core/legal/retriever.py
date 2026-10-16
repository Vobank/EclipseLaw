"""
Поиск статей под вопрос пользователя (ТЗ 5.4, этап 4).

Один способ поиска не закрывает все случаи, поэтому их три и они
дополняют друг друга:

* точный по номеру — «что говорит статья 1.3 УК»: смысловой поиск на
  такой вопрос выдаёт что угодно, потому что сам номер смысла не несёт;
* смысловой — «что грозит, если ударил человека»: в вопросе нет ни
  одного слова из текста статьи, и полнотекстовый поиск бессилен;
* полнотекстовый — страховка: вопрос задан дословной формулировкой,
  а модель эмбеддингов её не узнала.

Результаты сливаются ранговым слиянием (RRF): важно не насколько
уверен каждый способ по своей шкале — шкалы у них разные и несравнимые, —
а на каком месте он поставил статью.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from core.legal.db import LegalDB, StoredRule
from core.legal.query import (is_liability_question, is_list_question,
                              is_procedure_question, normalize)
from core.legal.vectors import EmbeddingError, VectorIndex

log = logging.getLogger(__name__)

# «статья 1.3», «ст. 2.1», «пункт 6.26», «п. 1.7.1»
NUMBER_IN_QUERY = re.compile(
    # Падежи целиком: «статьям», «статьях», «статьёй». Без «м» и «х»
    # перечисление «по статьям 6.1 и 12.8» проверка не видела, и
    # первый номер проходил мимо сверки.
    r"(?:стать[ияюеймхё]+|ст\.?|пункт[ауеов]?|п\.)\s*№?\s*"
    r"(\d+(?:\.\d+)*)",
    re.IGNORECASE)

# Вживую номер называют и без слова «статья»: «42 дк», «17.1 ук»,
# «что там по 12.3 административного». Слово «статья» при этом
# опускают почти всегда, и без этого шаблона точный поиск не срабатывал.
# Номера в перечислении после одного слова «статьям»: «6.1 и 12.8».
NUMBER_LIST = re.compile(
    r"(?:стать[ияюеймхё]+|пункт\w*|п\.)\s*№?\s*"
    r"\d+(?:\.\d+)*(?:\s*(?:,|и)\s*\d+(?:\.\d+)*)+")


def _numbers_in_list(text: str) -> list[str]:
    """Все номера из перечисления, а не только первый."""
    found: list[str] = []
    for chunk in NUMBER_LIST.findall(text):
        for number in re.findall(r"\d+(?:\.\d+)*", chunk):
            if number not in found:
                found.append(number)
    return found


NUMBER_WITH_CODE = re.compile(
    r"(\d+(?:\.\d+)*)\s*(?:-?[иы]?[йх]?\s*)?"
    r"(?:дк|ук|ак|ск|тк|пк|эк|вк|кодекс\w*|"
    r"дорожн\w*|уголовн\w*|административн\w*|судебн\w*|трудов\w*|"
    r"процессуальн\w*|этическ\w*|воздушн\w*)",
    re.IGNORECASE)

# Сокращение кодекса в вопросе -> как документ называется в базе.
# Нужно, чтобы «42 дк» не притащил статью 42 из другого кодекса.
# Номер, названный вообще без пояснений: «12.7», «что такое 12.8».
# Требуем точку и не больше трёх цифр в части, а также запрещаем соседство
# с цифрой, точкой и знаком валюты — иначе «штраф до 5.000$» превратится
# в запрос статьи 5.000.
NUMBER_BARE = re.compile(r"(?<![\d.,])(\d{1,3}(?:\.\d{1,3})+)(?![\d.,$])")

CODE_ALIASES = {
    "дк": "Дорожный", "дорожн": "Дорожный",
    "ук": "Уголовный", "уголовн": "Уголовный",
    "ак": "Административный", "административн": "Административный",
    "ск": "Судебный", "судебн": "Судебный",
    "тк": "Трудовой", "трудов": "Трудовой",
    "пк": "Процессуальный", "процессуальн": "Процессуальный",
    "эк": "Этический", "этическ": "Этический",
    "вк": "Воздушный", "воздушн": "Воздушный",
}


def _document_priority(title: str) -> int:
    """Чем меньше, тем выше документ в выдаче при одинаковом номере.

    Номер вроде «12.8» встречается в десятке документов сразу. Когда
    кодекс не назван, человек почти наверняка имеет в виду закон, а не
    внутреннее положение о баллах лидеров — по нему и сортируем.
    """
    lowered = title.lower()
    if "кодекс" in lowered or "конституция" in lowered:
        return 0
    if lowered.startswith("закон"):
        return 1
    if "устав" in lowered or "положение" in lowered:
        return 2
    return 3


def _procedure_priority(title: str) -> int:
    """Порядок документов для вопроса о правах и порядке действий.

    Здесь главный не Уголовный кодекс, а Процессуальный: человек
    спрашивает не состав преступления, а кто что обязан сделать.
    Конституция рядом — из неё берутся сами права задержанного.
    """
    lowered = title.lower()
    if "процессуальн" in lowered or "конституц" in lowered:
        return 0
    # Ордера и судопроизводство — второй эшелон по тем же вопросам:
    # «имеет ли право обыскать» решается Законом об ордерах, «как
    # обжаловать» — Судебным кодексом.
    if "ордер" in lowered or "судебн" in lowered or "неприкосновен" in lowered:
        return 1
    # Прочие законы сюда не попадают намеренно. Поднимать их скопом
    # нельзя: на «имеет ли право обыскать машину» первым вылезал
    # Закон «Об охоте» — там тоже про осмотр и изъятие, но не про то.
    return 2


def _base_number(number: str) -> str:
    """Номер без пометки о повторе: «64.1 (2)» -> «64.1».

    Повтор — наша внутренняя пометка, а не часть нормы: на форуме
    статья называется «64.1», и модель пишет в ответе именно так.
    Без снятия пометки проверка объявляла верную ссылку выдуманной,
    а вычёркивание стирало её из ответа.
    """
    return re.sub(r"\s*\(\d+\)$", "", number).strip()


def _duplicate_key(rule: StoredRule) -> str:
    """Ключ, по которому пункт считается копией другого.

    Сравниваем по тексту, а не по номеру: одна и та же норма лежит в
    уставе LSPD под 22.16 и в уставе шерифов под 23.13. Регистр и
    пунктуация не в счёт — при копировании с форума они гуляют.
    """
    return re.sub(r"[^а-яёa-z0-9 ]+", "", rule.text.lower()).strip()[:160]


def _collapse_duplicates(hits: list[Hit]) -> list[Hit]:
    """Оставляет одну копию нормы, остальные вешает на неё списком.

    В базе 148 текстов, повторяющихся в разных документах, — суммарно
    202 лишние копии. Без схлопывания на вопрос про халатность четыре
    из шести мест в выдаче занимали «халатно относиться к амуниции» и
    «беспорядок на парковке» из уставов LSPD и шерифов: один и тот же
    текст, просто в двух документах. Места в выдаче и в контексте
    модели уходили впустую.

    Выживает лучший по очкам — он уже отсортирован выше.

    Порог длины низкий намеренно. Сначала он был 40 символов, чтобы
    «Запрещено.» из разных документов не слиплось в одно, — но на
    вопросе о сроке действия пропуска четыре места из пяти заняла
    строка «Срок действия пропуска.» из пяти одинаковых тем про разные
    структуры. Если текст совпадает буква в букву, показывать его
    пятикратно бессмысленно независимо от длины: информации в нём
    столько же, сколько в одной копии.
    """
    best: dict[str, Hit] = {}
    result: list[Hit] = []
    for hit in hits:
        key = _duplicate_key(hit.rule)
        if len(key) < 12:
            result.append(hit)
            continue
        keeper = best.get(key)
        if keeper is None:
            best[key] = hit
            result.append(hit)
        else:
            keeper.duplicates.append(hit.rule.citation())
    return result


def _code_hint(question: str) -> str:
    """Какой кодекс имеется в виду, если он назван сокращением.

    Сокращение обязано быть отдельным словом: без этого «сколько стоит
    лечение» опознавалось как Судебный кодекс, потому что «ск» нашлось
    внутри «сколько».
    """
    lowered = question.lower()
    for alias, document in CODE_ALIASES.items():
        pattern = rf"\b{alias}\b" if len(alias) <= 2 else rf"\b{alias}"
        if re.search(pattern, lowered):
            return document
    return ""

# Константа рангового слияния. 60 — общепринятое значение: оно сглаживает
# разницу между первым и вторым местом, не давая одному способу поиска
# единолично определять выдачу.
RRF_K = 60

# Насколько поиск по фрагментам слабее остальных источников. Фрагменты
# есть только у длинных статей, и без штрафа они вытесняли короткие
# нормы, которые и были ответом.
CHUNK_RANK_OFFSET = 12

# Признаки перечня внутри статьи: «1) Губернатор; 2) Вице-Губернатор».
# На вопрос «кто относится к неприкосновенным лицам» отвечает именно
# такая статья, а не определение термина рядом с ней — а по смыслу
# определение к вопросу ближе, и перечень раз за разом проигрывал.
_ENUMERATION = re.compile(r"(?:^|\s)(?:\d{1,2}\)|[а-яё]\))\s+\S")
LIST_BONUS = 2.2


@dataclass
class Hit:
    rule: StoredRule
    score: float
    sources: list[str]
    # Пункты с тем же текстом в других документах. Они не занимают
    # отдельных мест в выдаче, но упоминаются: пользователю важно
    # знать, что норма продублирована в его уставе.
    duplicates: list[str] = field(default_factory=list)
    # Часть длинной статьи, которой она и нашлась. В контекст модели
    # идёт именно она: статья на пять тысяч знаков иначе доходила
    # обрезанной ровно по тому месту, где начинался ответ.
    fragment: str = ""
    # id фрагмента, которым статья нашлась. Текст по нему берётся
    # позже: ширина окна зависит от лимита на статью в режиме.
    chunk_id: int | None = None

    def why(self) -> str:
        return "+".join(self.sources)


class Retriever:
    def __init__(self, db: LegalDB, index: VectorIndex | None = None,
                 max_context_chars: int = 6000,
                 max_article_chars: int = 900,
                 slang: dict | None = None, stopwords=None,
                 document_bonus: list[float] | None = None,
                 usage_weight: float | None = None, combos=None):
        self.db = db
        self.index = index
        self.slang = slang
        self.stopwords = stopwords
        # Множитель по типу документа для вопросов о наказании, по
        # индексу _document_priority: кодекс, закон, устав, прочее.
        # Именно множитель, а не прибавка: прибавка одинаково поднимала
        # все кодексы разом, и в выдаче про халатность выше уставов
        # оказывались заодно хулиганство и неуплата налогов. Множитель
        # усиливает уже найденное, а не уравнивает подходящее с чужим.
        self.document_bonus = document_bonus or [1.6, 1.3, 1.0, 1.0]
        # Насколько поднимать документы, которыми реально пользуются.
        # Ровно 0 отключает учёт. Величина намеренно скромная: частота
        # помогает выбрать между двумя одинаково подходящими статьями,
        # но не должна вытаскивать наверх привычный документ вместо
        # точного попадания в редкий.
        self.usage_weight = 0.15 if usage_weight is None else usage_weight
        # Связки слов: «пристрелил» плюс «полицейский» значит совсем
        # не то же, что «пристрелил» само по себе.
        self.combos = combos
        self._usage: dict[str, float] | None = None
        self.max_context_chars = max_context_chars
        # Потолок на одну статью. Отдельные статьи уставов тянут на
        # несколько тысяч символов и в одиночку съедают весь контекст.
        self.max_article_chars = max_article_chars
        # Причина, по которой последний поиск шёл без смыслового
        # слоя. Пустая строка — всё в порядке.
        self.last_degraded = ""

    @classmethod
    def from_config(cls, cfg, db: LegalDB, index: VectorIndex | None = None):
        """Сборка по конфигу — чтобы утилиты искали ровно так же, как бот.

        Иначе проверка в tools/ показывает одну выдачу, а приложение
        отдаёт другую, и настройка словаря сленга идёт вслепую.
        """
        return cls(
            db, index,
            max_context_chars=cfg.get("retrieval.max_context_chars", 6000),
            max_article_chars=cfg.get("retrieval.max_article_chars", 900),
            slang=cfg.get("retrieval.slang"),
            stopwords=cfg.get("retrieval.stopwords"),
            document_bonus=cfg.get("retrieval.document_bonus"),
            usage_weight=cfg.get("retrieval.usage_weight"),
            combos=cfg.get("retrieval.combos"))

    def _usage_factor(self, title: str) -> float:
        """Множитель по тому, как часто документом реально пользуются.

        Логарифм, а не сами обращения: между документом с одним
        обращением и с двумя разница важна, между сотней и двумястами —
        уже нет. Иначе Уголовный кодекс, который спрашивают каждый
        день, через месяц начал бы перевешивать точное попадание
        в редкий закон.
        """
        if not self.usage_weight:
            return 1.0
        if self._usage is None:
            import math
            raw = self.db.usage_weights()
            top = max(raw.values()) if raw else 0
            scale = math.log1p(top) or 1.0
            self._usage = {t: math.log1p(n) / scale for t, n in raw.items()}
        return 1.0 + self.usage_weight * self._usage.get(title, 0.0)

    def forget_usage(self) -> None:
        """Сбрасывает кэш весов — после того, как их обновили."""
        self._usage = None

    # ------------------------------------------------------------------ поиск
    def retrieve(self, question: str, limit: int = 6) -> list[Hit]:
        ranked: dict[int, dict] = {}
        self.last_degraded = ""
        # id пункта -> id фрагмента, которым он нашёлся.
        fragments: dict[int, int] = {}

        def add(rule: StoredRule, position: int, source: str) -> None:
            entry = ranked.setdefault(
                rule.id, {"rule": rule, "score": 0.0, "sources": []})
            entry["score"] += 1.0 / (RRF_K + position)
            if source not in entry["sources"]:
                entry["sources"].append(source)

        # 1. Точный номер, если он назван в вопросе — хоть со словом
        # «статья», хоть в живой форме «42 дк», хоть просто «12.8».
        numbers = set(NUMBER_IN_QUERY.findall(question))
        numbers |= set(NUMBER_WITH_CODE.findall(question))
        numbers |= set(NUMBER_BARE.findall(question))
        hint = _code_hint(question)

        for number in numbers:
            found = self.db.get_rule(number)
            if hint:
                # Названный кодекс выигрывает решительно, а не на очко:
                # пункт 12.8 есть и в Уголовном кодексе, и в правилах
                # подачи объявлений, и в уставе FIB. Без этого нужная
                # статья оказывалась третьей, потому что чужие пункты
                # набирали очки ещё и смысловым поиском.
                named = [r for r in found if r.document.startswith(hint)]
                others = [r for r in found if not r.document.startswith(hint)]
                for rule in named:
                    add(rule, 0, "номер")
                    ranked[rule.id]["score"] += 1.0
                found = others
            else:
                # Кодекс не назван — вперёд идут кодексы и законы.
                found.sort(key=lambda r: _document_priority(r.document))
                if found:
                    # Одного лишь первого места мало: смысловой поиск
                    # добавляет очки другим документам с тем же номером,
                    # и «17.1» уходило в правила подачи объявлений вместо
                    # Уголовного кодекса. Поэтому лучший по типу документ
                    # получает прибавку, а не просто первую позицию.
                    add(found[0], 0, "номер")
                    ranked[found[0].id]["score"] += 0.5
                    found = found[1:]
            for position, rule in enumerate(found):
                add(rule, position, "номер")

        # Номера искали по исходному вопросу — там они и названы.
        # Смысловой и полнотекстовый поиск идут уже по вычищенному:
        # «что там за халатность будет» размывает вектор настолько,
        # что статья «Халатность» Уголовного кодекса выпадает из
        # первой пятёрки, уступая «халатно относиться к амуниции».
        search_text = normalize(question, self.slang, self.stopwords,
                                self.combos)

        # 2. Смысловой.
        if self.index is not None:
            try:
                for position, (rule_id, _, chunk_id) in enumerate(
                        self.index.search_detailed(
                            search_text, limit=limit * 3)):
                    rule = self.db.rule_by_id(rule_id)
                    if rule is not None:
                        add(rule, position, "смысл")
                        if chunk_id is not None:
                            fragments[rule.id] = chunk_id
            except EmbeddingError as exc:
                # Молча деградировать нельзя: без смыслового поиска
                # ответы заметно портятся, а человек видит только то,
                # что бот «вдруг стал хуже». Отметку забирает
                # оркестратор и показывает её в окне.
                log.warning("Смысловой поиск недоступен: %s", exc)
                self.last_degraded = str(exc)

        # 2б. Полнотекстовый по фрагментам — только как уточнение.
        # Он не вводит в выдачу новые статьи: на «обозву мента» куски
        # закона о LSPD обгоняли статью 17.3 Уголовного кодекса просто
        # потому, что у длинной статьи фрагменты есть, а у короткой
        # нормы — нет. Новые статьи по-прежнему приводит смысловой
        # поиск по фрагментам, а этот говорит, какую часть показать.
        if self.index is not None:
            try:
                for position, (rule_id, chunk_id) in enumerate(
                        self.index.search_chunks_text(
                            search_text, limit=limit * 2)):
                    if rule_id not in ranked:
                        continue
                    # Статья уже найдена — дословное совпадение внутри
                    # неё это подтверждает и поднимает её выше.
                    add(ranked[rule_id]["rule"],
                        position + CHUNK_RANK_OFFSET, "фрагмент")
                    fragments.setdefault(rule_id, chunk_id)
            except Exception as exc:
                log.warning("Поиск по фрагментам недоступен: %s", exc)

        # 3. Полнотекстовый.
        for position, rule in enumerate(
                self.db.search(search_text, limit=limit * 3)):
            add(rule, position, "текст")

        # 4. Поправка на тип документа. Раньше она работала только при
        # поиске по номеру, и на вопрос «что будет за оскорбление
        # полицейского» сверху оказывался внутренний устав LSPD —
        # дисциплинарное взыскание для сотрудника, а не статья для
        # гражданина. Когда спрашивают про наказание, кодекс важнее.
        if is_liability_question(question):
            for entry in ranked.values():
                entry["score"] *= self.document_bonus[
                    _document_priority(entry["rule"].document)]
        elif is_list_question(question):
            # Спросили перечень — наверх идут статьи, в которых он есть.
            for entry in ranked.values():
                if len(_ENUMERATION.findall(entry["rule"].text or "")) >= 3:
                    entry["score"] *= LIST_BONUS

            # Перечень бывает разнесён по отдельным пунктам: типы
            # ордеров — это 4.2 «1) AR», 4.3 «2) AS», 4.4, 4.5. В ответ
            # тогда попадали два типа из четырёх. Добираем остальные
            # и ставим их сразу за тем, который нашёлся.
            # Якорь ищем среди лучших, а не только в первом: перечень
            # редко выходит на первое место — обычно выше оказывается
            # статья с определением того же термина.
            best = next(
                (e for e in sorted(ranked.values(),
                                   key=lambda e: -e["score"])[:4]
                 if re.match(r"^\s*\d{1,2}\)", e["rule"].text or "")), None)
            if best is not None:
                anchor = best["score"]
                for shift, sibling in enumerate(
                        self.db.list_siblings(best["rule"])):
                    if sibling.id in ranked:
                        ranked[sibling.id]["score"] = max(
                            ranked[sibling.id]["score"], anchor * 0.99)
                    else:
                        ranked[sibling.id] = {
                            "rule": sibling,
                            "score": anchor * (0.99 - shift * 0.001),
                            "sources": ["перечень"]}
        elif is_procedure_question(question):
            # Тот же множитель, но другой порядок документов: на «меня
            # задержали, что делать» нужен Процессуальный кодекс, а не
            # состав преступления и не устав чужой фракции.
            for entry in ranked.values():
                entry["score"] *= self.document_bonus[
                    _procedure_priority(entry["rule"].document)]

        # 5. Поправка на то, чем реально пользуются. Считается по
        # документам, на которые ссылались готовые ответы: нашлось
        # шесть статей, пригодилась одна — вот её документ и в счёт.
        if self.usage_weight:
            for entry in ranked.values():
                entry["score"] *= self._usage_factor(entry["rule"].document)

        hits = []
        for e in ranked.values():
            hit = Hit(e["rule"], e["score"], e["sources"])
            # Сам текст фрагмента соберём при сборке контекста: там
            # известен лимит на статью, а значит и сколько соседних
            # кусков поместится рядом.
            hit.chunk_id = fragments.get(e["rule"].id)
            hits.append(hit)
        hits.sort(key=lambda h: h.score, reverse=True)
        return _collapse_duplicates(hits)[:limit]

    # ------------------------------------------------------------- контекст
    def _fragment(self, chunk_id: int | None, budget: int = 0) -> str:
        """Текст фрагмента по его id — или пусто, если статья короткая."""
        if chunk_id is None or self.index is None:
            return ""
        try:
            return self.index.chunk_text(chunk_id, budget)
        except Exception:      # индекс не обязан быть живым ради ответа
            log.debug("Фрагмент %s не прочитан", chunk_id, exc_info=True)
            return ""

    def context_budget(self, num_ctx: int, num_predict: int,
                       prompt_chars: int) -> int:
        """Сколько символов выдержек влезет в окно модели.

        Окно у профиля своё, и выдержки — единственная часть промпта,
        которая растёт. Замер: промпт экзамена с выдержками занял 4141
        токен при окне 4096, и Ollama отвечала 400 — экзамен в игровом
        профиле просто не работал. Считаем бюджет заранее.

        Считаем два символа выдержек на свободный токен, а не три:
        выдержки тянут за собой список разрешённых статей, и промпт
        растёт примерно в полтора раза от их объёма. Замер: 5730
        символов выдержек дали 2736 токенов, то есть 2.1 символа на
        токен. Округляем в свою пользу.
        """
        if not num_ctx:
            return self.max_context_chars
        # Что уже занято: сам промпт режима без выдержек, ответ модели
        # и запас на вопрос с разметкой диалога.
        reserved = num_predict + prompt_chars // 3 + 256
        free_tokens = num_ctx - reserved
        if free_tokens <= 0:
            return 0
        return min(self.max_context_chars, free_tokens * 2)

    def build_context(self, hits: list[Hit],
                      max_article_chars: int = 0,
                      max_context_chars: int = 0) -> str:
        """Собирает выдержки для промпта, укладываясь в лимит символов.

        Лимит нужен не ради красоты: на CPU-профиле окно модели 8192
        токена, и раздутый контекст вытесняет сам вопрос.

        Длинную статью обрезаем, а не выбрасываем вместе с остатком.
        Раньше первая же статья в 4000 символов упирала счётчик в лимит,
        цикл прерывался, и из двадцати двух статей чек-листа в промпт
        попадали три — модель оставалась без нужной нормы и сочиняла её.
        """
        # Потолок на статью можно поднять для конкретного режима:
        # на экзамене спрашивают полноту перечня, и статья, обрезанная
        # на середине перечисления, стоит балла.
        article_cap = max_article_chars or self.max_article_chars
        total_cap = max_context_chars or self.max_context_chars

        blocks: list[str] = []
        used = 0
        for hit in hits:
            if used >= total_cap:
                break
            block = hit.rule.full_text()
            if len(block) > article_cap:
                fragment = getattr(hit, "fragment", "") or self._fragment(
                    getattr(hit, "chunk_id", None), article_cap)
                if fragment and fragment not in block[:article_cap]:
                    # Обрезать с начала здесь нельзя: ответ на вопрос
                    # про вызов прокурора лежит в седьмом примечании
                    # статьи 3.4, то есть за четыре тысячи знаков от
                    # её начала. Берём ту часть, которой она нашлась.
                    head = block[:block.index(hit.rule.text)]                         if hit.rule.text in block else ""
                    block = (f"{head}[фрагмент статьи] {fragment}"
                             )[:max(article_cap, len(fragment) + 200)]
                else:
                    block = block[:article_cap].rstrip() + "..."
            if getattr(hit, "duplicates", None):
                # Та же норма в другом документе. Модель должна знать,
                # что сослаться можно и на неё: пользователь служит в
                # конкретной структуре, и ему нужен его устав.
                block += ("\nТа же норма продублирована в: "
                          + "; ".join(hit.duplicates[:3]))
            if used + len(block) > total_cap:
                continue      # не влезла эта — может влезть следующая
            blocks.append(block)
            used += len(block)
        return "\n\n".join(blocks)

    def articles_mentioned(self, hits: list[Hit]) -> list[str]:
        return [f"{h.rule.document}, п. {h.rule.number}" for h in hits]

    # Число с единицей измерения: «1.000$», «50 км/ч», «60 мин», «6 лет».
    # Именно такие величины пользователь уносит в игру, и именно их
    # модель охотнее всего подставляет по памяти.
    # Между числом и единицей закон вставляет оговорку: «41 и более
    # км/ч», «3 или более лет». Без её учёта выдержка давала только
    # «4000$», ответ — «41км/ч», и верная цифра вычёркивалась из
    # ответа как выдуманная.
    FIGURE = re.compile(
        # Год («2026 года», «в 2026 году») — не срок и не сумма.
        # Проверка ругалась на «2026года» как на выдуманную величину.
        r"(?<!\d)(?!\d{4}\s*год)(\d[\d\s.,]*?)"
        r"\s*(?:(?:и|или)\s+(?:более|менее|выше|ниже)\s+)?"
        r"(\$|км/ч|минут|мин|лет|года|год|часов|часа|час)"
        r"(?![а-яё])", re.IGNORECASE)

    @staticmethod
    def _figures(text: str) -> set[str]:
        found = set()
        for number, unit in Retriever.FIGURE.findall(text):
            # «1.000» и «1 000» — то же самое, что «1000».
            digits = re.sub(r"[\s.,]", "", number)
            if digits:
                found.add(f"{digits}{unit.lower()}")
        return found

    @staticmethod
    def unsupported_figures(answer: str, hits: list[Hit]) -> list[str]:
        """Суммы и сроки из ответа, которых нет в выдержках.

        Проверка номеров статей не ловит самое частое враньё: номер
        назван верный, а штраф и срок подставлены по памяти. Живой
        пример: на вопрос о скорости в городе модель выдала «штраф до
        1.000$ ((от 30 до 60 мин))» — сумма из статьи, а игровое время
        скопировано из примера в самом промпте, где оно стоит образцом
        формата. В базе у этой статьи никаких минут нет вовсе.

        Такую подмену пользователь не заметит: выглядит она ровно как
        настоящая санкция.

        Сверяем со статьями, на которые ответ СОСЛАЛСЯ, а не со всей
        выдачей. Разница принципиальная: в выдаче шесть статей из разных
        документов, и «30 минут» найдётся в какой-нибудь посторонней —
        проверка промолчит, хотя санкция приписана совсем другой норме.
        """
        cited = Retriever.cited_rules(answer, hits)
        rules = cited or [h.rule for h in hits]

        known: set[str] = set()
        for rule in rules:
            known |= Retriever._figures(
                " ".join([rule.text, rule.sanction, " ".join(rule.notes)]))

        unsupported: list[str] = []
        for figure in Retriever._figures(answer):
            if figure not in known and figure not in unsupported:
                unsupported.append(figure)
        return sorted(unsupported)

    @staticmethod
    def cited_rules(answer: str, hits: list[Hit]) -> list:
        """Статьи, на которые ответ реально сослался, в порядке упоминания.

        Нужно, чтобы дописать под ответом короткую расшифровку каждой
        названной статьи. Берём только те, что подавались модели:
        выдуманные номера ловит unsupported_citations, и расшифровывать
        их нечем.
        """
        by_number = {}
        for hit in hits:
            by_number.setdefault(hit.rule.number, hit.rule)
            # И под номером без пометки о повторе: в ответе модель
            # пишет «64.1», а в базе пункт хранится как «64.1 (2)».
            by_number.setdefault(_base_number(hit.rule.number), hit.rule)

        # Ищем не «номер вообще», а каждый конкретный номер из выдачи.
        # Общий шаблон здесь не годится: в базе есть «III.3» и «IV.1.1»,
        # римские цифры он не ловит, а ослабление шаблона притащило бы
        # из текста ответа «в течение 48 часов» и «до 5.000$».
        placed: list[tuple[int, object]] = []
        for number, rule in by_number.items():
            # Хвост «(?:\.\d+)*» — на случай, когда модель уточнила
            # подпункт «6.2.1» там, где подавалась статья «6.2».
            # Хвост запрещаем только перед цифрой: точка и запятая после
            # номера — обычная пунктуация («смотрите 1.2.», «4.7, III.2»),
            # а вот «5.0» внутри «5.000$» ловить не нужно.
            pattern = (r"(?<![\w.])" + re.escape(number)
                       + r"(?:\.\d{1,3})*(?![\d$])(?!\.\d)")
            if "." not in number:
                # Голая «1» встретится в любой фразе про сроки и суммы,
                # поэтому для неё требуем явную ссылку: «ст. 1», «п. 1».
                pattern = r"(?:стать[ияюей]+|ст\.?|пункт[ауе]?|п\.)\s*№?\s*" + pattern
            match = re.search(pattern, answer, re.IGNORECASE)
            if match is not None:
                placed.append((match.start(), rule))

        placed.sort(key=lambda pair: pair[0])
        return [rule for _, rule in placed]

    # --------------------------------------------------------- проверка ответа
    @staticmethod
    def unsupported_citations(answer: str, hits: list[Hit]) -> list[str]:
        """Номера статей, названные в ответе, но отсутствующие в выдержках.

        Полагаться на то, что модель послушает запрет выдумывать, нельзя:
        на проверке qwen3:4b-instruct к вопросу о представлении при
        задержании сочинил «Статья 14.1, штраф 500 долларов», хотя ни
        того, ни другого в контексте не было. Промпт — просьба, а это
        проверка: сверяем каждый названный номер с тем, что реально
        подавалось модели.
        """
        allowed = {h.rule.number for h in hits}
        allowed |= {_base_number(n) for n in allowed}
        # Номер статьи допустим и в виде «1.3.1» при поданном «1.3»:
        # модель уточняет подпункт, а не выдумывает норму.
        prefixes = {p for number in allowed
                    for p in _number_prefixes(number)}

        # Ищем оба написания. В документах номер ставят перед кодексом
        # («6.1 УК SA»), и шаблон со словом «статья» его не видел:
        # прокурорское дело ушло с двумя неверными статьями, а проверка
        # промолчала.
        numbers = list(NUMBER_IN_QUERY.findall(answer))
        numbers += [n for n in NUMBER_WITH_CODE.findall(answer)
                    if n not in numbers]
        # Перечисление: «по статьям 6.1 и 12.8», «п. 2.1, 2.2».
        # Слово «статья» стоит один раз, и без этого шаблона все
        # номера кроме первого проходили мимо сверки.
        numbers += [n for n in _numbers_in_list(answer)
                    if n not in numbers]

        unsupported: list[str] = []
        for number in numbers:
            if number in allowed or number in prefixes:
                continue
            # Уточнение подпункта засчитываем только от составного
            # номера: «1.3» -> «1.3.1» это та же норма. От голой
            # цифры — нет: в выдержках попадается статья «6», и через
            # неё в документ проходила любая «6.x». Так в прокурорское
            # дело и попала статья 6.1 о телесных повреждениях за то,
            # что сотрудник не зачитал права.
            if any("." in a and number.startswith(f"{a}.") for a in allowed):
                continue
            if number not in unsupported:
                unsupported.append(number)
        return unsupported


def _number_prefixes(number: str) -> list[str]:
    parts = number.split(".")
    return [".".join(parts[:i]) for i in range(1, len(parts) + 1)]
