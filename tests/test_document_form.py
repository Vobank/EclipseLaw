"""
Выбор бумаги в режиме оформления.

Раньше вид документа угадывался по надиктовке. Но факты дела одни и
те же, а бумаг из них пишется несколько разных: определение о принятии
иска, о переносе заседания, приговор, решение. Судейских образцов почти
три десятка — выбрать между ними может только человек.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import Config  # noqa: E402
from core.documents import KINDS, TemplateStore  # noqa: E402
from core.prompts import document_prompt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def store() -> TemplateStore:
    return TemplateStore(Config.load().resolve_path("storage.templates_dir"))


def test_court_forms_are_listed(store):
    forms = dict(store.forms("court"))
    assert forms, "судебные образцы не разобраны по видам"
    assert any("ОПРЕДЕЛЕНИЕ" in name for name in forms)


def test_same_form_counts_its_samples(store):
    """Трёх «определений о рассмотрении ходатайства» должно быть три."""
    forms = dict(store.forms("court"))
    assert max(forms.values()) >= 2


def test_forms_are_sorted_by_frequency(store):
    counts = [count for _, count in store.forms("court")]
    assert counts == sorted(counts, reverse=True)


def test_personal_file_names_are_not_forms(store):
    """Образцы дел названы по людям («Lukas Marlboro 15.09.2026») —
    вида документа в таком имени нет."""
    assert store.forms("proceedings") == []


# ------------------------------------------------------------- промпт
def test_chosen_form_reaches_the_prompt():
    prompt = document_prompt("образцы", "скелет", "надиктовка",
                             form="ОПРЕДЕЛЕНИЕ о принятии искового заявления")
    assert "ОПРЕДЕЛЕНИЕ о принятии искового заявления" in prompt
    assert "не подменяй его" in prompt


def test_without_a_choice_prompt_is_unchanged():
    assert "ОФОРМЛЯЕМ ИМЕННО ЭТУ БУМАГУ" not in document_prompt(
        "образцы", "скелет", "надиктовка")


# --------------------------------------------------------- подключение
def test_orchestrator_asks_before_dictation():
    source = (ROOT / "core" / "orchestrator.py").read_text(encoding="utf-8")
    assert "_ask_document_form" in source
    assert "sig_form.connect(self._choose_document_form)" in source


def test_chosen_form_picks_the_samples():
    """Иначе модели показывают приговор, когда пишется определение."""
    source = (ROOT / "core" / "orchestrator.py").read_text(encoding="utf-8")
    assert 'like=f"{form} {dictation}" if form else dictation' in source


def test_overlay_reports_the_choice():
    source = (ROOT / "ui" / "overlay.py").read_text(encoding="utf-8")
    assert 'href.startswith("form:")' in source
