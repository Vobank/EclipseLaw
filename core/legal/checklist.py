"""
Чек-лист процессуальных требований к задержанию (ТЗ 5.3, 5.4).

Список не выдуман и не зашит в код текстом: каждый пункт — это тема,
по которой в базе ищется реальная статья. Поэтому чек-лист остаётся
верным после любого обновления кодекса, а ссылки в вопросах адвоката
указывают на то, что действительно есть в базе.

Темы перечислены в config.yaml, секция lawyer.checklist — их правят
без изменения кода.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# Темы по умолчанию. Выведены из реального Процессуального кодекса
# сервера: презумпция невиновности, правило Миранды, порядок ареста.
DEFAULT_TOPICS: tuple[dict, ...] = (
    {"key": "основание",
     "title": "Основание задержания",
     "query": "основания для задержания подозреваемого инициировать следствие"},
    {"key": "права",
     "title": "Зачитывание прав (правило Миранды)",
     "query": "права задержанного зачитаны до допроса недопустимые "
              "доказательства правило Миранды"},
    {"key": "представился",
     "title": "Сотрудник представился",
     "query": "сотрудник обязан представиться назвать должность при "
              "обращении к гражданину"},
    {"key": "адвокат",
     "title": "Право на адвоката",
     "query": "право задержанного на адвоката защитника юридическую помощь"},
    {"key": "допрос",
     "title": "Порядок и лимиты допроса",
     "query": "порядок проведения допроса подозреваемого количество сотрудников "
              "время допроса не может превышать"},
    {"key": "обыск",
     "title": "Основание для обыска и досмотра",
     "query": "обыск досмотр личный осмотр задержанного основания"},
    {"key": "презумпция",
     "title": "Презумпция невиновности",
     "query": "презумпция невиновности бремя доказывания вины"},
    {"key": "видео",
     "title": "Видеофиксация",
     "query": "видеофиксация запись задержания доказательства"},
    {"key": "арест",
     "title": "Порядок ареста и выбор учреждения",
     "query": "процедура ареста объявить о доказательствах доставить "
              "к месту отбытия наказания"},
)


@dataclass
class ChecklistItem:
    key: str
    title: str
    # Готовая фраза, которую подзащитный произносит вслух. Заготовлена,
    # а не сочиняется моделью: одиннадцать вопросов на CPU-профиле
    # считались бы минут десять, а нужны они в первую же секунду
    # задержания. {ссылка} подставляется из найденной статьи.
    question: str = ""
    # Каким должен быть ответ сотрудника, если процедура соблюдена.
    # Нужен не меньше самого вопроса: без него человек слышит ответ
    # и не понимает, всё ли в нём на месте, а нарушение выглядит так
    # же убедительно, как соблюдение.
    expected: str = ""
    articles: list = field(default_factory=list)
    covered: bool = False

    @property
    def has_basis(self) -> bool:
        return bool(self.articles)

    def citations(self) -> str:
        return "; ".join(a.citation() for a in self.articles)

    def spoken_question(self) -> str:
        """Фраза для произнесения, с подставленной ссылкой на статью."""
        if not self.question:
            return ""
        if not self.articles:
            return self.question.replace("{ссылка}", "").strip()
        rule = self.articles[0]
        # Название документа в скобках, а не в родительном падеже:
        # склонять «Закон "Об адвокатуре и адвокатской деятельности"»
        # по-русски пришлось бы отдельным кодом, а в скобках фраза
        # читается вслух без запинки и остаётся грамматически верной.
        reference = f"статье {rule.number} ({_short(rule.document)})"
        return self.question.replace("{ссылка}", reference)


    def as_prompt_block(self) -> str:
        lines = [f"— {self.title}"]
        if self.question:
            lines.append(f"   Готовый вопрос: «{self.spoken_question()}»")
        for article in self.articles:
            lines.append(f"   [{article.citation()}] {article.text[:400]}")
        if not self.articles:
            lines.append("   (в базе не нашлось статьи по этой теме — "
                         "спрашивать о ней без ссылки нельзя)")
        return "\n".join(lines)


def _mentions(text: str, number: str) -> bool:
    """Назван ли в тексте именно этот пункт.

    Проверка подстрокой годилась ровно до первого совпадения внутри
    другого номера: «6.2» находится и в «16.2», и в «6.22», и пункт
    чек-листа отмечался пройденным, хотя речь шла о другой статье.
    Отмеченный по ошибке пункт адвокат уже не задаст.
    """
    pattern = r"(?<![\w.])" + re.escape(number) + r"(?:\.\d{1,3})*(?![\d])(?!\.\d)"
    return re.search(pattern, text) is not None


def _short(title: str) -> str:
    """Короткое название документа для произнесения вслух."""
    return re.sub(r"\s*\(ред\.[^)]*\)", "", title).strip()


@dataclass
class Checklist:
    items: list[ChecklistItem] = field(default_factory=list)

    @property
    def grounded(self) -> list[ChecklistItem]:
        """Только пункты, под которые нашлась реальная статья."""
        return [i for i in self.items if i.has_basis]

    def as_prompt(self) -> str:
        blocks = [i.as_prompt_block() for i in self.grounded]
        return "\n".join(blocks)

    def as_overlay_text(self) -> str:
        """Готовые вопросы — то, что подзащитный произносит вслух."""
        lines = []
        for index, item in enumerate(self.grounded, 1):
            mark = "x" if item.covered else " "
            lines.append(f"[{mark}] {index}. {item.title}")
            spoken = item.spoken_question()
            if spoken:
                lines.append(f"      «{spoken}»")
            if item.expected:
                lines.append(f"      ждём в ответ: {item.expected}")
            if item.articles:
                lines.append(f"      [{item.articles[0].citation()}]")
            lines.append("")
        missing = [i.title for i in self.items if not i.has_basis]
        if missing:
            lines.append("Без опоры в базе (спрашивать без ссылки нельзя): "
                         + ", ".join(missing))
        return "\n".join(lines)

    def next_question(self) -> ChecklistItem | None:
        """Первый непройденный пункт — с него и продолжать."""
        return next((i for i in self.grounded if not i.covered), None)

    def mark_covered(self, text: str) -> list[str]:
        """Отмечает пункты, чьи статьи уже прозвучали в диалоге."""
        just_covered = []
        for item in self.grounded:
            if item.covered:
                continue
            if any(_mentions(text, a.number) for a in item.articles):
                item.covered = True
                just_covered.append(item.title)
        return just_covered


def build_checklist(retriever, topics=None, per_topic: int = 2) -> Checklist:
    """Собирает чек-лист, подбирая под каждую тему статьи из базы.

    Берём по одной-две статьи на тему, а не по шесть: чек-лист целиком
    уходит в системный промпт, и на CPU-профиле окно модели всего
    8192 токена.
    """
    checklist = Checklist()
    for topic in (topics or DEFAULT_TOPICS):
        try:
            hits = retriever.retrieve(topic["query"], limit=per_topic)
        except Exception:
            log.exception("Не удалось подобрать статьи по теме %s", topic["key"])
            hits = []
        checklist.items.append(ChecklistItem(
            key=topic["key"],
            title=topic["title"],
            question=topic.get("question", ""),
            expected=topic.get("expected", ""),
            articles=[h.rule for h in hits[:per_topic]],
        ))
    log.info("Чек-лист собран: %d пунктов, с опорой в базе %d",
             len(checklist.items), len(checklist.grounded))
    return checklist
