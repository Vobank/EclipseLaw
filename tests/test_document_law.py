"""
Сверка документов с законодательной базой.

Документы писались вообще без базы: ни один номер статьи не
проверялся. Проверка на живом деле показала, чем это кончается —
за «не зачитал права» модель поставила УК 6.1 (нанесение телесных
повреждений), за отсутствие видеозаписи УК 12.8 (оборот оружия).
Номера настоящие, статьи не те, и в готовом деле это заметит
только юрист.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import Config  # noqa: E402
from core.legal.retriever import Retriever  # noqa: E402
from core.prompts import document_prompt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


class Rule:
    def __init__(self, number, document="Уголовный Кодекс"):
        self.id = id(self)
        self.number, self.document = number, document
        self.text = "текст нормы"
        self.sanction = ""


class Hit:
    def __init__(self, rule):
        self.rule = rule


# ------------------------------------------------------------- промпт
def test_law_extracts_reach_the_prompt():
    prompt = document_prompt("обр", "скелет", "надиктовка",
                             law="[УК, п. 15.1] Превышение полномочий.")
    assert "СТАТЬИ ИЗ ЗАКОНОДАТЕЛЬНОЙ БАЗЫ" in prompt
    assert "15.1" in prompt


def test_extracts_go_after_the_samples():
    """Сначала они стояли в начале: между ними и заданием вклинивались
    четыре тысячи знаков образцов, и модель брала номера по памяти."""
    prompt = document_prompt("ОБРАЗЕЦ_МЕТКА", "скелет", "надиктовка",
                             law="ВЫДЕРЖКА_МЕТКА")
    assert prompt.index("ОБРАЗЕЦ_МЕТКА") < prompt.index("ВЫДЕРЖКА_МЕТКА")


def test_charges_are_handed_to_the_model():
    prompt = document_prompt("обр", "скелет", "надиктовка",
                             charges="  Уголовный Кодекс, п. 15.1 — превышение")
    assert "СТАТЬИ ОБВИНЕНИЯ" in prompt
    assert "15.1" in prompt


def test_charges_come_last():
    """Список обвинения короткий и должен остаться перед глазами."""
    prompt = document_prompt("обр", "скелет", "надиктовка",
                             law="ВЫДЕРЖКА_МЕТКА", charges="ОБВИНЕНИЕ_МЕТКА")
    assert prompt.index("ВЫДЕРЖКА_МЕТКА") < prompt.index("ОБВИНЕНИЕ_МЕТКА")


# --------------------------------------------------------- сверка номеров
def test_number_before_the_code_is_checked():
    """«6.1 УК SA» — обычное написание в документах, и оно проходило
    мимо проверки: та знала только формат «статья 6.1»."""
    hits = [Hit(Rule("15.1")), Hit(Rule("15.6"))]
    assert "6.1" in Retriever.unsupported_citations("6.1 УК SA – нарушение",
                                                    hits)


def test_case_forms_are_understood():
    hits = [Hit(Rule("15.1"))]
    found = Retriever.unsupported_citations("по статьям 6.1 и 12.8", hits)
    assert "6.1" in found and "12.8" in found


def test_bare_number_does_not_whitelist_its_children():
    """В выдержках попадается статья «6», и через неё проходила любая
    «6.x» — так в дело и попала 6.1 о телесных повреждениях."""
    hits = [Hit(Rule("6"))]
    assert "6.1" in Retriever.unsupported_citations("6.1 УК SA", hits)


def test_real_subitem_still_passes():
    """«1.3» подан, модель уточнила «1.3.1» — это та же норма."""
    hits = [Hit(Rule("1.3"))]
    assert Retriever.unsupported_citations("по статье 1.3.1", hits) == []


def test_correct_article_passes():
    hits = [Hit(Rule("15.1"))]
    assert Retriever.unsupported_citations("по статье 15.1 УК SA", hits) == []


# ------------------------------------------------------------- настройки
def test_liability_searches_are_configured():
    """Поиск по надиктовке приносил только процессуальные нормы —
    как положено задерживать, — а статьи ответственности не попадали."""
    cfg = Config.load()
    always = " ".join(cfg.get("documents.always_search") or [])
    assert "полномочий" in always or "халатность" in always
    assert cfg.get("documents.charge_search")


def test_orchestrator_searches_the_base_for_documents():
    source = (ROOT / "core" / "orchestrator.py").read_text(encoding="utf-8")
    assert "_document_law" in source and "_charges" in source


def test_document_is_rewritten_on_a_bad_article():
    source = (ROOT / "core" / "orchestrator.py").read_text(encoding="utf-8")
    assert "DOCUMENT_RETRY" in source


def test_warnings_are_split_by_severity():
    """Неверная статья рушит дело, непроверенный срок — повод сверить."""
    source = (ROOT / "core" / "orchestrator.py").read_text(encoding="utf-8")
    assert "СТАТЬИ НЕТ В БАЗЕ" in source
    assert "СВЕРЬТЕ ЦИФРЫ" in source


def test_year_is_not_a_figure():
    """«22 октября 2026 года» — не срок и не сумма."""
    assert Retriever._figures("22 октября 2026 года") == set()
    assert "6лет" in Retriever._figures("от 3 до 6 лет")


# ------------------------------------------------- что видно на экране
def test_screen_is_synced_after_redaction():
    """Документ печатается по мере генерации: вычеркнутое имя судьи
    осталось бы на экране, хотя в тексте уже стоит [НЕ УКАЗАНО]."""
    source = (ROOT / "core" / "orchestrator.py").read_text(encoding="utf-8")
    assert "self.overlay.sig_replace.emit(shown, document)" in source


def test_rewritten_document_replaces_the_first():
    """Иначе человек читает два документа подряд."""
    source = (ROOT / "core" / "orchestrator.py").read_text(encoding="utf-8")
    assert "[статья не сошлась с базой — переписываю]" in source
    assert "sig_replace.emit(\n                        document," in source


def test_redaction_note_says_it_was_removed():
    """«ВЫЧЕРКНУТО: имя Tyler Tea» читалось так, будто бот его написал."""
    source = (ROOT / "core" / "orchestrator.py").read_text(encoding="utf-8")
    assert "УБРАЛ ИЗ ДОКУМЕНТА" in source
    assert "на их месте стоит [НЕ УКАЗАНО]" in source


def test_status_returns_after_model_switch():
    """Возврат быстрой модели ставил «Загружаю...» поверх готового
    документа — бот выглядел думающим, хотя уже закончил."""
    source = (ROOT / "core" / "orchestrator.py").read_text(encoding="utf-8")
    restore = source[source.index("def _restore_fast_model"):]
    restore = restore[:restore.index("\n    def ", 10)]
    assert "finally:" in restore and "_status(" in restore
