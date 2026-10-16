"""
Две роли режима адвоката и слух за экзаменатором.

Задержали меня самого — вопросы задаются от первого лица. Приехал
к задержанному как частный адвокат — вопросы те же по сути, но задаёт
их защитник, и первым делом ему нужно то, чего у самого задержанного
нет: допуск к подзащитному, встреча наедине, адвокатская тайна.

Отдельно — вопрос экзаменатора. Его речь идёт тем же путём, что и
любая реплика собеседника: захват по клавише, распознавание, ответ.
Повторять вопрос голосом не нужно.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import Config  # noqa: E402
from core.prompts import system_prompt  # noqa: E402
from core.session import Mode, Source  # noqa: E402


@pytest.fixture(scope="module")
def cfg() -> Config:
    return Config.load()


# ------------------------------------------------------------------ роли
def test_both_roles_are_configured(cfg):
    roles = cfg.get("lawyer.roles")
    assert set(roles) == {"detained", "counsel"}


def test_each_role_has_its_own_questions(cfg):
    roles = cfg.get("lawyer.roles")
    detained = {item["key"] for item in roles["detained"]["checklist"]}
    counsel = {item["key"] for item in roles["counsel"]["checklist"]}
    assert detained != counsel
    # То, что есть только у защитника: встреча наедине и адвокатская
    # тайна. «Допуск» и «вызов» убраны — адвоката пускают и так.
    assert {"наедине", "тайна", "опрос"} <= counsel


def test_counsel_questions_speak_about_the_client(cfg):
    """Защитник говорит «мой подзащитный», а не «моё задержание»."""
    counsel = cfg.get("lawyer.roles")["counsel"]["checklist"]
    texts = " ".join(item["question"] for item in counsel)
    assert "подзащитн" in texts
    assert "моего задержания" not in texts


def test_detained_questions_speak_in_first_person(cfg):
    detained = cfg.get("lawyer.roles")["detained"]["checklist"]
    texts = " ".join(item["question"] for item in detained)
    assert "моего задержания" in texts


def test_every_question_cites_an_article(cfg):
    """Без ссылки вопрос сотруднику не работает — его просто отклонят."""
    for role in cfg.get("lawyer.roles").values():
        for item in role["checklist"]:
            assert "{ссылка}" in item["question"], item["key"]
            assert item.get("query"), item["key"]


def test_default_role_is_the_usual_one(cfg):
    assert cfg.get("lawyer.default_role") == "detained"


# ---------------------------------------------------------------- промпт
def test_counsel_is_the_state_advocate():
    """Пользователь — государственный адвокат, а не частный.

    Разница не косметическая: частный работает по договору с клиентом
    и может освобождать под залог, государственного вызывает сам
    сотрудник, когда задержанный требует защитника. Вопросы про договор
    и залог государственному адвокату задавать нечего.
    """
    cfg = Config.load()
    counsel = cfg.get("lawyer.roles")["counsel"]
    assert "государственный" in counsel["title"].lower()
    texts = " ".join(item["question"] for item in counsel["checklist"]).lower()
    assert "залог" not in texts
    assert "договор" not in texts
    assert "лицензи" not in texts


def test_counsel_prompt_says_no_contract_needed():
    prompt = system_prompt(Mode.LAWYER, legal_context="выдержка",
                           rag_enabled=True, lawyer_role="counsel")
    assert "ГОСУДАРСТВЕННЫЙ адвокат" in prompt
    assert "залог" in prompt      # сказано, что это НЕ его право


def test_counsel_prompt_switches_the_role():
    prompt = system_prompt(Mode.LAWYER, legal_context="выдержка",
                           rag_enabled=True, lawyer_role="counsel")
    assert "Задержан не пользователь, а его подзащитный" in prompt


def test_detained_prompt_stays_first_person():
    prompt = system_prompt(Mode.LAWYER, legal_context="выдержка",
                           rag_enabled=True, lawyer_role="detained")
    assert "Задержан не пользователь" not in prompt
    assert "произносит сам задержанный" in prompt


def test_role_does_not_leak_into_other_modes():
    prompt = system_prompt(Mode.EXAM, legal_context="выдержка",
                           rag_enabled=True, lawyer_role="counsel")
    assert "Задержан не пользователь" not in prompt


# ------------------------------------------------- вопрос экзаменатора
def test_interlocutor_speech_is_answered_not_stored():
    """Речь собеседника уходит в ответ во всех режимах, кроме документа.

    На экзамене это и есть «услышать экзаменатора»: нажал клавишу,
    он задал вопрос, бот ответил — повторять за ним не нужно.
    """
    import inspect

    from core.orchestrator import Orchestrator

    source = inspect.getsource(Orchestrator._run_pipeline)
    # Единственная ветка, которая не отвечает, — режим документа.
    assert "Mode.DOCUMENT" in source
    assert "self._answer(transcript.text, source)" in source


def test_other_source_exists():
    assert Source.OTHER.title
