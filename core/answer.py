"""
Генерация ответа с проверкой ссылок на статьи.

Промпт — это просьба, а не гарантия. Модель на четыре миллиарда
параметров нарушает прямой запрет выдумывать регулярно, и хуже всего
то, как она это делает: на вопрос о правах задержанного она выдаёт
складный перечень из настоящего права — молчание, звонок, адвокат —
и подписывает его номером, которого в базе нет вовсе. Такой ответ
выглядит убедительнее правильного, и заметить подмену можно только
по номеру.

Поэтому проверка здесь детерминированная: каждый названный номер
сверяется с тем, что реально подавалось модели. Не сошлось — модель
получает свой промах в лицо и переписывает ответ. Срабатывает редко,
поэтому на обычную скорость не влияет.

Обёртка одна на всех: и бот, и проверочные утилиты зовут её, иначе
tools/ меряет одно, а пользователь получает другое.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from core.llm.ollama_client import LLMUnavailable, Message
from core.prompts import STRICTER_RETRY

log = logging.getLogger(__name__)


def redact_invented(text: str, numbers: list[str]) -> str:
    """Вычёркивает из ответа ссылки на статьи, которых нет в базе.

    Последний рубеж, когда и перегенерация не помогла. Пересказ модель
    всё равно оставит свой — с этим ничего не сделать, — но фальшивый
    номер до пользователя не дойдёт, а именно он придаёт выдумке вид
    нормы, на которую можно сослаться в игре или на экзамене.

    Проверено на правах задержанного: модель дважды подряд подписала
    перечень из настоящего права пунктом «Процессуальный Кодекс, 5.1»,
    которого в базе нет вовсе.
    """
    for number in numbers:
        escaped = re.escape(number)
        # Ссылка целиком: «[Процессуальный Кодекс, п. 5.1]».
        text = re.sub(r"\[[^\[\]]*?\bп\.\s*" + escaped + r"\s*\]",
                      "[ССЫЛКА НЕ ПОДТВЕРЖДЕНА]", text)
        # И упоминание в тексте: «в статье 5.1», «согласно п. 5.1».
        # Точка после номера — обычная пунктуация конца предложения
        # («в статье 5.1.»), а вот «5.0» внутри «5.000$» трогать нельзя.
        text = re.sub(r"(?:стать[ияюей]+|ст\.|пункт[ауе]?|п\.)\s*№?\s*"
                      + escaped + r"(?![\d])(?!\.\d)",
                      "(статья не подтверждена)", text, flags=re.IGNORECASE)
    return text


def check_answer(retriever, answer: str, hits: list) -> list[str]:
    """Всё, что в ответе названо, но в выдержках отсутствует.

    Номера статей и величины проверяются вместе: назвать верную статью
    и приписать ей чужой штраф — ошибка не меньшая, чем выдумать номер,
    а заметить её пользователю ещё труднее.
    """
    numbers = retriever.unsupported_citations(answer, hits)
    figures = retriever.unsupported_figures(answer, hits)
    return numbers + [f for f in figures if f not in numbers]


@dataclass
class CheckedAnswer:
    text: str
    # Номера, которых нет в выдержках, — после всех попыток.
    invented: list[str] = field(default_factory=list)
    # Пришлось ли переписывать ответ. Нужно для отчётов: рост этого
    # числа означает, что модель пора менять на более крупную.
    retried: bool = False


def generate_checked(llm, messages: list[Message], retriever, hits: list,
                     *, num_predict: int = 0, retry: bool = False,
                     redact: bool = False, model: str = "",
                     on_chunk=None, on_retry=None) -> CheckedAnswer:
    """Генерирует ответ и, если названы чужие статьи, просит переписать.

    on_chunk вызывается на каждый кусок потока — для показа в оверлее.
    on_retry(номера, забракованный_ответ) вызывается один раз перед
    повторной попыткой: показать причину и убрать из ленты то, что уже
    напечатано, иначе второй ответ допишется следом.
    """
    def run() -> str:
        pieces: list[str] = []
        for chunk in llm.stream_chat(messages, num_predict=num_predict,
                                     model=model):
            pieces.append(chunk)
            if on_chunk is not None:
                on_chunk(chunk)
        return "".join(pieces).strip()

    answer = run()
    invented = check_answer(retriever, answer, hits)
    if not invented or not retry:
        return CheckedAnswer(answer, invented, retried=False)

    log.warning("Перегенерация: статьи вне выдержек %s", invented)
    if on_retry is not None:
        # Отдаём и забракованный ответ: он уже напечатан в ленте, и
        # без него вызывающий не сможет его оттуда убрать — второй
        # ответ просто допишется следом, и человек увидит два подряд.
        on_retry(invented, answer)

    # Промах показываем модели отдельным сообщением, а не правкой
    # системного промпта: конкретный номер в диалоге работает заметно
    # лучше общего напоминания в начале.
    messages = messages + [
        Message("assistant", answer),
        Message("user", STRICTER_RETRY.format(numbers=", ".join(invented))),
    ]
    try:
        second = run()
    except LLMUnavailable as exc:
        log.error("Перегенерация не удалась: %s", exc)
        return CheckedAnswer(answer, invented, retried=True)

    if not second:
        second, still = answer, invented
    else:
        still = check_answer(retriever, second, hits)

    # Не помогла и вторая попытка — вычёркиваем ссылку. Выдумка
    # останется, но без номера она хотя бы не выглядит нормой.
    if still and redact:
        log.warning("Ссылки вычеркнуты как неподтверждённые: %s", still)
        second = redact_invented(second, still)
    return CheckedAnswer(second, still, retried=True)


# Раздел заключения, в котором обязан стоять состав, а не процедура.
_LIABILITY_SECTION = re.compile(
    r"СТАТЬЯ\s+ДЛЯ\s+ПРИВЛЕЧЕНИЯ[^\n]*\n?(.*?)(?=\n\s*\**\s*\d\s*[.)]|\Z)",
    re.IGNORECASE | re.DOTALL)

# Признаки должностного состава: по ним из найденного подбирается то,
# что вообще можно предъявить сотруднику.
_MISCONDUCT = re.compile(
    r"превышени[ея]\s+долж|злоупотреблени|халатност|бездействи[ея]|"
    r"должностн(?:ых|ые)\s+полномочи|неисполнени[ея]\s+(?:своих\s+)?"
    r"(?:должностных\s+)?обязанност", re.IGNORECASE)


def wrong_attribution(retriever, verdict: str, hits: list) -> str:
    """Проверяет, что сотрудника привлекают по составу, а не по процедуре.

    Просьбы в промпте здесь не хватило: модель раз за разом писала
    «привлечь по статье 6.2 Процессуального кодекса» — а это норма
    о недопустимых доказательствах, ответственности она не
    устанавливает, и требование по ней суд отклонит. Состав при этом
    лежал в тех же выдержках.

    Возвращает готовую приписку или пустую строку, если всё в порядке.
    """
    section = _LIABILITY_SECTION.search(verdict)
    if section is None:
        return ""
    cited = retriever.cited_rules(section.group(1), hits)
    if not cited:
        return ""
    if any("процессуальн" not in (rule.document or "").lower()
           for rule in cited):
        return ""        # хоть один нормальный состав назван — довольно

    available = [rule for rule in (h.rule for h in hits)
                 if _MISCONDUCT.search(rule.text or "")
                 and "процессуальн" not in (rule.document or "").lower()]
    if not available:
        return ("\n\n[!] ПРОВЕРЬТЕ: привлечь по Процессуальному кодексу "
                "нельзя — он устанавливает порядок, а не ответственность. "
                "Подходящего состава среди найденных статей нет.")

    listed = "; ".join(
        f"{rule.document}, п. {rule.number}" for rule in available[:3])
    return ("\n\n[!] ПРОВЕРЬТЕ: привлечь по Процессуальному кодексу нельзя — "
            "он устанавливает порядок, а не ответственность. Состав есть "
            f"здесь: {listed}.")


# Строка перечня со ссылкой на конце: «- Ордер AR даёт право... [Закон
# "Об ордерах", п. 4.2]». Двоеточие перед скобкой модель ставит сама.
_LIST_LINE = re.compile(
    r"^(\s*[-—•]\s*)(.+?)[\s:：]*\[([^\],]+),\s*п\.\s*([^\]]+)\]\s*$")


def fold_citations(text: str) -> str:
    """Сводит ссылки перечня в одну строку под списком.

    Промпта тут не хватило: когда пункты перечня взяты из разных
    пунктов одного закона (типы ордеров — это 4.2, 4.3, 4.4, 4.5),
    модель ставит ссылку в каждую строку, и список читается как
    таблица. Смысл при сворачивании не меняется — меняется только
    то, сколько места занимает ответ.

    Если документы разные, всё остаётся как было: тогда ссылка в
    строке несёт смысл.
    """
    lines = text.split("\n")
    marked = [(index, _LIST_LINE.match(line))
              for index, line in enumerate(lines)]
    hits = [(index, match) for index, match in marked if match]
    if len(hits) < 2:
        return text

    documents = {match.group(3).strip() for _, match in hits}
    if len(documents) != 1:
        return text

    numbers: list[str] = []
    for index, match in hits:
        lines[index] = f"{match.group(1)}{match.group(2).rstrip(' :：')}"
        for number in match.group(4).split(","):
            number = number.strip()
            if number and number not in numbers:
                numbers.append(number)

    document = documents.pop()
    tail = f"[{document}, п. {', '.join(numbers)}]"
    # Ставим ссылку сразу за последним пунктом списка: дальше в ответе
    # может идти ещё текст, и ссылка в самом конце к нему не относится.
    lines.insert(hits[-1][0] + 1, tail)
    return "\n".join(lines)


# Служебные блоки, которые дописывает сама программа. Модель однажды
# приняла их за образец своего ответа и стала печатать сама — в ленте
# выходило два одинаковых блока подряд. Контекст диалога мы почистили,
# но подстраховка нужна: промпт — это просьба, а не гарантия.
_SERVICE_BLOCK = re.compile(
    r"\n*──\s*(?:О ЧЁМ СТАТЬЯ|СТАТЬЯ ЦЕЛИКОМ|СКОРЕЕ ВСЕГО ЭТА СТАТЬЯ)"
    r"\s*──.*?(?=\n\s*\n|\Z)", re.IGNORECASE | re.DOTALL)

_SERVICE_TAIL = re.compile(
    r"\n*Текст статьи целиком\s*—\s*ctrl\+alt\+\w+\s*$", re.IGNORECASE)


def strip_service_blocks(text: str) -> str:
    """Убирает из ответа модели то, что дописывает сама программа."""
    cleaned = _SERVICE_BLOCK.sub("", text)
    cleaned = _SERVICE_TAIL.sub("", cleaned)
    return cleaned.rstrip()


# Ссылка на статью в квадратных скобках.
_CITATION = re.compile(r"\[[^\]\n]{2,120}?,\s*п\.\s*[^\]\n]{1,24}\]")


def dedupe_citations(text: str) -> str:
    """Оставляет каждую ссылку один раз — последним упоминанием.

    Модель повторяет номер и в тексте, и в скобках, и в конце: «по
    основаниям, указанным в п. 2.1 (2) статьи [ПК, п. 2.1 (2)].
    [ПК, п. 2.1 (2)]». Читать это невозможно, а произнести вслух —
    тем более. Порядок ссылок сохраняем: убираем ранние повторы,
    последний оставляем на месте.
    """
    found = _CITATION.findall(text)
    if len(found) < 2:
        return text

    seen_total: dict[str, int] = {}
    for citation in found:
        seen_total[citation] = seen_total.get(citation, 0) + 1

    used: dict[str, int] = {}

    def keep(match: "re.Match[str]") -> str:
        citation = match.group()
        used[citation] = used.get(citation, 0) + 1
        # Оставляем только последнее вхождение каждой ссылки.
        if used[citation] < seen_total[citation]:
            return ""
        return citation

    cleaned = _CITATION.sub(keep, text)
    # После вычистки остаются висячие пробелы и «статьи .» — подчищаем.
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    cleaned = re.sub(r"\bстатьи\s*\.", "", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()
