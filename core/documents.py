"""
Генератор документов: делопроизводство и судопроизводство (ТЗ 5.5, этап 8).

Два набора шаблонов держатся раздельно и не смешиваются: структура
прокурорского дела и структура судебного документа разные, и подмешать
одну в другую — значит получить бумагу, которую не примут.

Образцы лежат файлами в data/templates/<вид>/ и подаются модели как
few-shot примеры. Их можно добавлять и убирать без изменения кода:
чем больше образцов, тем ближе результат к принятому на сервере стилю.

Главное ограничение из ТЗ: модель реструктурирует надиктованное, но
не добавляет ни одного факта сверх сказанного. Незаполненное
обязательное поле помечается [НЕ УКАЗАНО], а не выдумывается: в
прокурорском деле выдуманный номер паспорта или постановления — это
сфальсифицированный документ.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# Виды документов. Ключ — имя папки с образцами.
KINDS = {
    "proceedings": "Делопроизводство (прокурорское дело)",
    "court": "Судопроизводство (иск, жалоба, ходатайство)",
}

# Скелет прокурорского дела, выведенный из четырёх реальных образцов.
# Используется, когда образцов в папке ещё нет: даже без few-shot
# модель должна выдавать документ правильной структуры, а не свободный
# текст.
PROCEEDINGS_SKELETON = """\
\tпрокурор
<Имя прокурора>
  к Генеральному прокурору
к Заместителю Генерального прокурора

Статус дела: <открыто / закрыто>
Дата начала расследования: <ДД.ММ.ГГГГ>
Дата окончания расследования: <ДД.ММ.ГГГГ>

Фигуранты дела

Обвиняемый сотрудник <СТРУКТУРА>: <Имя Фамилия>
Паспорт: <номер>

Ход расследования

<Абзац 1: дата, принятие электронного обращения, от кого и на кого.>
<Абзац 2: изложение обращения со слов заявителя.>
<Абзац 3: ознакомление с личным делом, что установлено.>
<Абзац 4: возбуждение дела, постановление DJP-№<номер> об истребовании:
перечень истребуемого.>
<Абзац 5: исполнение постановления, что предоставлено.>
<Абзац 6: детальный анализ видеофиксации — что именно на ней видно.>
<Абзац 7: какие нормы нарушены при задержании или аресте.>
<Абзац 8: вывод прокуратуры об усмотрении нарушений.>

Материалы дела

<перечень приобщённого: обращение, постановления, видеофиксации,
кадровые аудиты, подтверждения уведомлений, данные из database.gov>

Предварительное заключение

Признать виновным сотрудника <СТРУКТУРА> <Имя>, номер паспорта <номер>,
по следующим статьям:

<статья> УК SA – <обоснование со ссылкой на нарушенную процессуальную
норму: статья, глава, пункты>

Окончательное решение

Признать виновным сотрудника <СТРУКТУРА> <Имя>, номер паспорта <номер>,
по статьям <перечень>. Избрать меру пресечения в виде <штрафной санкции
в размере N долларов / лишения свободы сроком на N лет>.

Дата: <ДД.ММ.ГГГГ>
Подпись:"""

SKELETONS = {"proceedings": PROCEEDINGS_SKELETON}


@dataclass
class Template:
    name: str
    kind: str
    text: str

    @property
    def preview(self) -> str:
        first = next((l.strip() for l in self.text.splitlines() if l.strip()), "")
        return first[:60]


@dataclass
class TemplateSet:
    kind: str
    title: str
    templates: list[Template] = field(default_factory=list)

    @property
    def skeleton(self) -> str:
        return SKELETONS.get(self.kind, "")

    def _closest(self, like: str, limit: int) -> list:
        """Образцы, больше всего похожие на надиктовку.

        Пока образцов было два, годился любой. Судья прислал двадцать
        девять, и среди них определения, решения, приговор и судебный
        ордер — структура у них разная. Показать модели приговор, когда
        диктуется определение, значит получить документ не того вида.

        Похожесть считается по общим словам: у документа одного вида с
        надиктовкой совпадут и вид, и обороты. Эмбеддинги тут излишни —
        образцов десятки, а не тысячи.
        """
        wanted = _significant_words(like)
        if not wanted:
            return self.templates[:limit]

        scored = []
        for template in self.templates:
            words = _significant_words(template.text[:1500])
            if not words:
                continue
            shared = len(wanted & words)
            scored.append((shared / len(wanted), template))
        scored.sort(key=lambda pair: -pair[0])
        best = [t for score, t in scored if score > 0][:limit]
        return best or self.templates[:limit]

    def few_shot(self, limit: int = 2, max_chars: int = 7000,
                 like: str = "") -> str:
        """Образцы для промпта.

        Берём не все: прокурорское дело — это две-три тысячи слов, и
        три таких образца вытеснят из окна модели саму надиктовку.

        like — надиктованный текст: по нему выбираются образцы того же
        вида, что и нужный документ.
        """
        chosen = self._closest(like, limit) if like else self.templates[:limit]
        blocks: list[str] = []
        used = 0
        for template in chosen:
            text = template.text
            if used + len(text) > max_chars:
                text = text[:max(0, max_chars - used)].rstrip() + "\n<...>"
            if not text.strip():
                break
            blocks.append(f"=== ОБРАЗЕЦ: {template.name} ===\n{text}")
            used += len(text)
        return "\n\n".join(blocks)


# Слова, по которым образец узнаётся как «того же вида». Служебные
# и общие для всех судебных бумаг отбрасываются: «суд», «штата»,
# «гражданин» есть в каждом документе и ничего не различают.
_COMMON_WORDS = frozenset("""
    суд суда суде судья судьи штата сан андреас лос сантос года
    гражданин гражданина далее статье статьи согласно настоящего
    данного которые также если после этом свои свое быть был была
""".split())


def _significant_words(text: str) -> set[str]:
    """Слова длиннее четырёх букв, кроме общих для всех документов."""
    words = re.findall(r"[а-яёa-z]{5,}", text.lower())
    return {w for w in words if w not in _COMMON_WORDS}


# Имя образца начинается с номера («07 РЕШЕНИЕ резолютивная...»), и
# сам заголовок бумаги стоит сразу за ним. Из него и берём название
# вида: «РЕШЕНИЕ», «ОПРЕДЕЛЕНИЕ о принятии искового заявления».
_FORM_NAME = re.compile(r"^\s*\d*\s*(.+?)\s*$")


def _form_name(stem: str) -> str:
    """Название бумаги из имени файла образца."""
    match = _FORM_NAME.match(stem)
    if not match:
        return ""
    name = match.group(1).strip()
    # Имена образцов делопроизводства — это имена людей и даты
    # («Lukas Marlboro 15.09.2026»), названия вида в них нет.
    if not any(ch.isupper() and ch.isalpha() and ch.isascii() is False
               for ch in name[:20]):
        return ""
    return name[:70]


class TemplateStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def directory(self, kind: str) -> Path:
        path = self.root / kind
        path.mkdir(parents=True, exist_ok=True)
        return path

    def load(self, kind: str) -> TemplateSet:
        if kind not in KINDS:
            raise ValueError(f"неизвестный вид документа: {kind}")
        result = TemplateSet(kind=kind, title=KINDS[kind])
        for path in sorted(self.directory(kind).glob("*.txt")):
            try:
                text = path.read_text(encoding="utf-8").strip()
            except Exception:
                log.exception("Не удалось прочитать образец %s", path)
                continue
            if text:
                result.templates.append(Template(path.stem, kind, text))
        log.info("Образцов «%s»: %d", result.title, len(result.templates))
        return result

    def add(self, kind: str, text: str, name: str = "") -> Path:
        """Сохраняет образец. Имя выводится из текста, если не задано."""
        if kind not in KINDS:
            raise ValueError(f"неизвестный вид документа: {kind}")
        text = text.replace("\r\n", "\n").strip()
        if len(text) < 200:
            raise ValueError(
                f"в образце всего {len(text)} символов — это не документ")

        if not name:
            name = _guess_name(text)
        safe = re.sub(r"[^\w\s-]", "", name).strip()[:60] or "образец"

        directory = self.directory(kind)
        path = directory / f"{safe}.txt"
        counter = 2
        while path.exists():
            path = directory / f"{safe} ({counter}).txt"
            counter += 1
        # newline="" обязателен: текст из буфера уже с CRLF, и Windows
        # иначе превратит их в CRCRLF, удвоив строки при чтении.
        path.write_text(text, encoding="utf-8", newline="")
        log.info("Образец сохранён: %s", path)
        return path

    def forms(self, kind: str) -> list[tuple[str, int]]:
        """Какие бумаги вообще есть среди образцов этого вида.

        Судейских образцов почти три десятка, и все они называются
        по-разному: определение о принятии иска, о переносе заседания,
        приговор, решение. Выбирать из них должен человек — бот по
        надиктовке угадывает только тему, но не то, какую именно
        бумагу вы сейчас пишете.

        Возвращает пары (название, сколько образцов), от частых к
        редким.
        """
        counts: dict[str, int] = {}
        for path in sorted(self.directory(kind).glob("*.txt")):
            name = _form_name(path.stem)
            if name:
                counts[name] = counts.get(name, 0) + 1
        return sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))

    def counts(self) -> dict[str, int]:
        return {kind: len(list(self.directory(kind).glob("*.txt")))
                for kind in KINDS}


@dataclass
class Requirement:
    """Что должно прозвучать в надиктовке, чтобы документ был полным."""
    key: str
    title: str
    hint: str                       # что именно сказать, если не хватает
    patterns: tuple[str, ...] = ()  # регулярки-признаки
    words: tuple[str, ...] = ()     # слова-признаки

    def satisfied(self, text: str) -> bool:
        lowered = text.lower()
        if any(w in lowered for w in self.words):
            return True
        return any(re.search(p, text, re.IGNORECASE) for p in self.patterns)


# Поля выведены из четырёх реальных прокурорских дел: без любого из них
# документ придётся дописывать руками.
PROCEEDINGS_REQUIREMENTS = (
    Requirement("dates", "Даты расследования",
                "назовите дату начала и дату окончания расследования",
                patterns=(r"\b\d{1,2}\.\d{2}\.\d{4}\b",
                          r"\b\d{1,2}\s+(январ|феврал|март|апрел|ма[йя]|"
                          r"июн|июл|август|сентябр|октябр|ноябр|декабр)"),),
    Requirement("applicant", "Заявитель",
                "кто подал обращение — имя и фамилия гражданина",
                words=("заявител", "обращение от", "гражданин")),
    Requirement("accused", "Обвиняемый сотрудник",
                "имя сотрудника и его структура (LSPD, FIB, GOV, EMS)",
                words=("lspd", "fib", "gov", "ems", "lssd", "saspa",
                       "сотрудник")),
    Requirement("passport", "Паспорт обвиняемого",
                "номер паспорта сотрудника",
                patterns=(r"паспорт\D{0,12}\d{4,}", r"\b\d{6}\b")),
    Requirement("badge", "Идентификационный знак",
                "номер жетона сотрудника, если он известен",
                patterns=(r"жетон\D{0,12}\d+", r"знак\D{0,12}\d+",
                          r"№\s*\d{2,4}\b")),
    Requirement("ordinance", "Постановление об истребовании",
                "номер постановления DJP и что именно вы истребовали",
                patterns=(r"постановлени\w*\D{0,20}\d{3,}", r"djp\D{0,6}\d+")),
    Requirement("essence", "Суть обращения",
                "что именно произошло со слов заявителя",
                words=("задержал", "задержание", "арест", "обыск",
                       "остановил", "применил", "требование")),
    Requirement("video", "Видеофиксация",
                "предоставлена видеофиксация или нет, и что на ней видно",
                words=("видеофиксац", "видео", "запись", "не сохранил")),
    Requirement("violation", "Что нарушено",
                "какие действия сотрудника вы считаете нарушением",
                words=("не представ", "не идентифиц", "не провёл",
                       "не провел", "нарушил", "превысил", "не соблюд",
                       "проигнорировал", "халатн")),
    Requirement("articles", "Статьи обвинения",
                "по каким статьям признать виновным — например 17.5 или 15.6",
                patterns=(r"\b\d{1,2}\.\d{1,2}\b",)),
    Requirement("measure", "Мера пресечения",
                "штраф и сумма либо срок лишения свободы",
                words=("штраф", "лишени", "долларов", "$", "увольнен",
                       "мера пресечен")),
)

COURT_REQUIREMENTS = (
    Requirement("court", "Куда подаётся",
                "в какой суд: Верховный, Апелляционный, Кассационный",
                words=("суд", "верховн", "апелляц", "кассацион")),
    Requirement("applicant", "Заявитель",
                "кто подаёт — имя, фамилия, номер паспорта",
                words=("заявител", "истец", "от гражданина", "паспорт")),
    Requirement("respondent", "Ответчик",
                "против кого подаётся — имя и структура",
                words=("ответчик", "сотрудник", "структур", "против")),
    Requirement("essence", "Обстоятельства",
                "что произошло, когда и где",
                words=("задержал", "арест", "обыск", "нарушил", "произошл")),
    Requirement("articles", "Правовое основание",
                "на какие статьи вы ссылаетесь",
                patterns=(r"\b\d{1,2}\.\d{1,2}\b",)),
    Requirement("claim", "Требование",
                "чего вы просите — снять статьи, компенсацию, отменить арест",
                words=("прошу", "требу", "снять", "компенсац", "отменить",
                       "признать")),
)

REQUIREMENTS = {
    "proceedings": PROCEEDINGS_REQUIREMENTS,
    "court": COURT_REQUIREMENTS,
}


@dataclass
class Advice:
    kind: str
    covered: list[Requirement] = field(default_factory=list)
    missing: list[Requirement] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        """Документ можно составлять: ключевое названо."""
        return len(self.missing) <= 2

    def as_text(self, dictated_words: int = 0) -> str:
        """Что названо, что осталось и что именно про это сказать.

        Раньше список шёл дважды: сверху названия пунктов, снизу три
        подсказки и строка «...и ещё 6 пунктов». Толку от неё не было:
        какие именно — человек не видел, а по названию
        «Идентификационный знак» не догадаешься, что от тебя хотят.
        Теперь подсказка стоит у своего пункта, и список один.
        """
        total = len(self.covered) + len(self.missing)
        lines = [f"НАДИКТОВАНО: {len(self.covered)} из {total} пунктов"]
        if dictated_words:
            lines[0] += f", слов {dictated_words}"
        lines.append("")

        for item in self.covered:
            lines.append(f"  [x] {item.title}")
        for item in self.missing:
            if item.hint:
                # Разделитель выбираем по подсказке: в одних уже есть
                # тире («кто подал — имя и фамилия»), в других
                # двоеточие («в какой суд: Верховный»). Берём тот знак,
                # которого в подсказке нет, иначе строка двоится.
                dash = "—" in item.hint
                separator = ":" if dash else " —"
                lines.append(f"  [ ] {item.title}{separator} {item.hint}")
            else:
                lines.append(f"  [ ] {item.title}")

        if not self.missing:
            lines.extend(["", "Всё основное названо — можно составлять."])
        return "\n".join(lines)


def advise(dictation: str, kind: str = "proceedings") -> Advice:
    """Что уже надиктовано и чего не хватает для полного документа.

    Проверка идёт признаками, а не моделью: подсказка нужна сразу после
    каждой реплики, а обращение к модели на CPU-профиле стоит минуты.
    Точность тут не критична — это напоминание, а не приговор.
    """
    advice = Advice(kind=kind)
    for requirement in REQUIREMENTS.get(kind, PROCEEDINGS_REQUIREMENTS):
        if requirement.satisfied(dictation):
            advice.covered.append(requirement)
        else:
            advice.missing.append(requirement)
    return advice


def invented_facts(document: str, dictation: str) -> list[str]:
    """Находит в документе точные данные, которых не было в надиктовке.

    Промпт запрещает их выдумывать, но 4B-модель запрет нарушает: на
    проверке она сама проставила «арестован в 14:15» и «уведомлён
    в 16:12», хотя ни одного времени продиктовано не было. Правдоподобно
    выглядящее время опаснее явного пробела — по нему считают
    процессуальные сроки, и перепроверять его никто не станет.

    Ищем только то, что обязано совпадать дословно: время, даты и
    длинные номера. Формулировки модель вправе менять, факты — нет.
    """
    patterns = (
        (r"\b\d{1,2}:\d{2}\b", "время"),
        (r"\b\d{1,2}\.\d{2}\.\d{4}\b", "дата"),
        (r"\b\d{6,}\b", "номер"),
        # Номер дела или иска: «№6950», «DJP-3505». Модель берёт
        # их прямо из образца — а это номер чужого дела.
        (r"№\s?\d{3,6}\b", "номер дела"),
        (r"\b[A-Z]{2,4}-[A-Z0-9-]{3,20}\b", "номер дела"),
        # Имя латиницей: «Tyler Tea». Первый же документ, показанный
        # судье, был подписан именем судьи из образца — чужим.
        (r"\b[A-Z][a-z]+\s+[A-Z][a-zA-Z]+\b", "имя"),
    )
    found: list[str] = []
    for pattern, kind in patterns:
        for value in re.findall(pattern, document):
            value = value.strip()
            if value in found:
                continue
            # Сравниваем без пробелов и решёток: в надиктовке
            # «жетон 417», в документе «№417» — один и тот же факт.
            flat = re.sub(r"[\s№]+", "", value).lower()
            said = re.sub(r"[\s№]+", "", dictation).lower()
            if flat in said or _is_known_term(value):
                continue
            found.append(f"{kind} {value}")
    return found


# Названия структур, судов и мест латиницей — не имена людей. Без
# этого списка вычёркивались бы «Los Santos» и «San Andreas»,
# которые стоят в любом документе по делу.
_KNOWN_TERMS = frozenset("""
los santos san andreas state prison authority police department sheriff
federal investigation bureau emergency medical service national guard
united states secret service weazel news government district court
supreme court appeal court blaine county paleto bay grand senora
sandy shores vespucci vinewood rockford hills la mesa el burro
""".split())


def _is_known_term(value: str) -> bool:
    """Название структуры или места, а не имя человека."""
    words = [w.strip(".,").lower() for w in value.split()]
    return all(word in _KNOWN_TERMS for word in words if word)


# Метка, которой промпт велит помечать незаполненное. Ею же заменяем
# выдуманное: пустое поле пользователь заполнит за десять секунд, а
# правдоподобное время, взятое моделью из головы, он может не заметить.
UNKNOWN = "[НЕ УКАЗАНО]"


def redact_invented_facts(document: str, dictation: str) -> tuple[str, list[str]]:
    """Вычёркивает из документа данные, которых не было в надиктовке.

    Предупреждения под документом мало: бумага уходит в дело целиком, и
    выдуманное время останется в ней, даже если приписка замечена.
    А в прокурорском деле подставленное время — это не опечатка, а
    сфальсифицированный документ: по нему считают процессуальные сроки.

    Заменяется ровно то, что обязано совпадать дословно, — время, даты
    и длинные номера. Формулировки модель вправе менять, факты нет.
    Проверено на qwen2.5:7b: он проставил «14:22», «16:15» и «23:42»
    в деле, где времени не диктовалось вовсе.
    """
    invented = invented_facts(document, dictation)
    if not invented:
        return document, []
    for item in invented:
        # Тип бывает из двух слов («номер дела»), поэтому режем по
        # первому пробелу с конца — иначе в замену уходило «дела
        # №6950» вместо «№6950», и номер оставался в документе.
        value = _fact_value(item)
        if not value:
            continue
        document = re.sub(rf"(?<!\d){re.escape(value)}(?!\d)",
                          UNKNOWN, document)
    return document, invented


# Типы фактов, которые ищет invented_facts. Нужны, чтобы отделить тип
# от значения: «номер дела DJP-3505» — это тип «номер дела».
_FACT_KINDS = ("номер дела", "время", "дата", "номер", "имя")


def _fact_value(item: str) -> str:
    """Само значение из строки «тип значение»."""
    for kind in _FACT_KINDS:
        if item.startswith(kind + " "):
            return item[len(kind) + 1:].strip()
    return item.split(" ", 1)[-1].strip()


def _guess_name(text: str) -> str:
    """Имя образца по содержимому: фамилия обвиняемого и дата."""
    accused = re.search(r"Обвиняемый[^:\n]*:\s*([^\n]+)", text)
    date = re.search(r"Дата начала расследования:\s*([\d.]+)", text)
    parts = []
    if accused:
        parts.append(accused.group(1).strip())
    if date:
        parts.append(date.group(1).strip())
    return " ".join(parts) if parts else "образец"


def unfilled_fields(document: str, limit: int = 8) -> list[str]:
    """Где именно в документе не хватает данных.

    «Полей без данных: 6» человеку ничего не говорит: найти их в
    готовой бумаге глазами тяжело, а что вписывать — непонятно.
    Поэтому называем каждое место: если перед меткой стоит название
    поля («Дата окончания расследования: [НЕ УКАЗАНО]»), берём его,
    а если метка попала в середину фразы — короткий кусок текста
    после неё, чтобы место узнавалось.
    """
    found: list[str] = []
    for line in document.split("\n"):
        if UNKNOWN not in line:
            continue
        for part in line.split(UNKNOWN)[:-1]:
            head = part.rstrip()
            # «Что-то: » перед меткой — это название поля.
            if head.endswith(":"):
                name = head.rstrip(":").strip()
                # Берём последнее предложение: в строке перед полем
                # бывает целый абзац.
                name = re.split(r"(?<=[.;])\s+", name)[-1].strip()
                if name:
                    found.append(name[:60])
                    continue
            # Текст после метки годится в подпись, только если он
            # осмысленный: когда там осталась одна точка, выходило
            # «перед «....»» — такая строка не помогает никому.
            tail = line.split(UNKNOWN, 1)[1].strip(" .,;:»«")
            if len(tail) >= 4:
                found.append(f"перед «{' '.join(tail.split()[:6])}...»")
                continue
            clean_head = head.strip(" .,;:»«")
            if len(clean_head) >= 4:
                found.append(f"после «...{' '.join(clean_head.split()[-6:])}»")

    # Повторы убираем: одна и та же дата часто встречается дважды.
    unique: list[str] = []
    for name in found:
        if name not in unique:
            unique.append(name)
    return unique[:limit]
