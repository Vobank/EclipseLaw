"""
Что всплыло на настоящем экзамене.

Три вещи: ответ иногда печатался дважды, язык был канцелярский —
такое вслух не произносят, — и ответ шёл слишком долго, потому что
карту занимала запущенная игра.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.answer import dedupe_citations  # noqa: E402
from core.config import Config  # noqa: E402
from core.gpu import choose_profile, free_vram_mb  # noqa: E402
from core.prompts import BY_MODE  # noqa: E402
from core.session import Mode  # noqa: E402


# --------------------------------------------------------- двойной ответ
def test_retry_hands_back_the_discarded_answer():
    """Забракованный ответ уже напечатан: без него второй допишется
    следом, и человек прочитает два ответа подряд."""
    source = (Path(__file__).resolve().parent.parent
              / "core" / "answer.py").read_text(encoding="utf-8")
    assert "on_retry(invented, answer)" in source


def test_orchestrator_erases_the_discarded_answer():
    source = (Path(__file__).resolve().parent.parent
              / "core" / "orchestrator.py").read_text(encoding="utf-8")
    assert "sig_replace.emit(discarded.strip(), note)" in source


# ------------------------------------------------------ повторы ссылок
def test_repeated_citation_is_collapsed():
    text = ("Задерживают по основаниям из [ПК, п. 2.1]. [ПК, п. 2.1]")
    assert dedupe_citations(text).count("[ПК, п. 2.1]") == 1


def test_last_citation_stays_in_place():
    """Ссылка должна остаться в конце, а не уехать в середину фразы."""
    text = "Смотри [ПК, п. 6.1] и ещё раз. [ПК, п. 6.1]"
    assert dedupe_citations(text).rstrip().endswith("[ПК, п. 6.1]")


def test_different_citations_all_survive():
    text = "Первое [УК, п. 1.1], второе [ПК, п. 2.2]."
    result = dedupe_citations(text)
    assert "[УК, п. 1.1]" in result and "[ПК, п. 2.2]" in result


def test_single_citation_is_untouched():
    text = "Обычный ответ. [Дорожный Кодекс, п. 32]"
    assert dedupe_citations(text) == text


# ------------------------------------------------------------- язык
def test_exam_prompt_bans_officialese():
    exam = BY_MODE[Mode.EXAM]
    assert "ВСЛУХ" in exam
    for word in ("осуществляется", "является", "данный"):
        assert word in exam, "в промпте нет списка канцелярских слов"


def test_exam_prompt_forbids_pointing_at_a_number():
    """«По основаниям, указанным в пункте 2.1» — это не ответ."""
    assert "НЕ ОТСЫЛАЙ К НОМЕРУ" in BY_MODE[Mode.EXAM]


# ---------------------------------------------------------- скорость
def test_profile_ladder_is_ordered_and_real():
    """Лестница профилей: сверху полный, снизу тот, что влезет всегда."""
    cfg = Config.load()
    ladder = cfg.get("hardware.memory_ladder")
    profiles = cfg.get("profiles")
    assert ladder, "лестницы нет — выбор профиля работать не будет"
    for step in ladder:
        assert step["profile"] in profiles, step
    needs = [step["needs_mb"] for step in ladder]
    assert needs == sorted(needs, reverse=True), "пороги идут не по убыванию"
    assert needs[-1] == 0, "последний профиль должен подходить всегда"


def test_recognition_never_degrades():
    """Распознавание — то место, где экономить нельзя: искажённый
    вопрос уводит ответ целиком, и никакая модель это не спасёт."""
    cfg = Config.load()
    for step in cfg.get("hardware.memory_ladder"):
        stt = cfg.get("profiles")[step["profile"]]["stt"]
        assert stt["model"] == "medium", step["profile"]
        assert stt["device"] == "cuda", step["profile"]


def test_explicit_profile_wins():
    """Человек назвал профиль руками — значит знает, что делает."""
    cfg = Config.load()
    chosen, reason = choose_profile(cfg, "heavy")
    assert chosen == "heavy"
    assert "вручную" in reason


def test_free_vram_is_a_number_or_none():
    value = free_vram_mb()
    assert value is None or (isinstance(value, int) and value >= 0)


def test_own_models_do_not_count_against_the_profile(monkeypatch):
    """Ollama держит модель два часа после выхода. Для nvidia-smi эта
    память занята, но она наша: при старте бот её освободит. Без
    поправки он видел «свободно 846 МБ» и сам понижал себе профиль —
    снаружи это выглядело как «бот стал хуже»."""
    import core.gpu as gpu

    cfg = Config.load()
    ladder = cfg.get("hardware.memory_ladder")
    full = ladder[0]["profile"]

    monkeypatch.setattr(gpu, "free_vram_mb", lambda: 846)
    monkeypatch.setattr(gpu, "ollama_held_mb", lambda host: 9922)
    chosen, reason = gpu.choose_profile(cfg)
    assert chosen == full
    assert "держит Ollama" in reason

    # Память заняла игра, а не мы — тогда понижение оправдано.
    monkeypatch.setattr(gpu, "ollama_held_mb", lambda host: 0)
    chosen, _ = gpu.choose_profile(cfg)
    assert chosen != full


def test_embedding_model_is_not_counted_as_reclaimable(monkeypatch):
    """bge-m3 никто не выгружает: без неё встаёт поиск по базе."""
    import core.gpu as gpu

    class Fake:
        @staticmethod
        def raise_for_status():
            pass

        @staticmethod
        def json():
            return {"models": [
                {"name": "bge-m3:latest", "size_vram": 660 * 1024 * 1024},
                {"name": "qwen2.5:7b-instruct", "size_vram": 4700 * 1024 * 1024},
            ]}

    monkeypatch.setattr(gpu.requests, "get", lambda *a, **k: Fake())
    assert gpu.ollama_held_mb("http://127.0.0.1:11434") == 4700


def test_exam_keeps_many_articles_rather_than_long_ones():
    """Важнее охват, чем длина одной статьи.

    Когда потолок на статью подняли до 3000, две-три длинные съедали
    весь бюджет: на «какие права у задержанного» статья 8, где они и
    перечислены, до модели не доходила. Узкий потолок на статью и
    широкий на весь контекст работают лучше.
    """
    cfg = Config.load()
    assert cfg.get("modes.exam.max_article_chars") <= 1500
    assert cfg.get("modes.exam.top_k") >= 12
    assert cfg.get("retrieval.max_context_chars") >= 9000


# ------------------------------------------------- выдержки и окно модели
@pytest.fixture(scope="module")
def retriever():
    from core.legal.db import LegalDB
    from core.legal.retriever import Retriever
    from core.legal.vectors import EmbeddingClient, VectorIndex

    cfg = Config.load()
    path = cfg.resolve_path("storage.legal_db")
    if not Path(path).exists():
        pytest.skip("базы законов нет")
    db = LegalDB(path)
    client = EmbeddingClient(cfg.get("llm.host"), "bge-m3")
    return Retriever.from_config(cfg, db, VectorIndex(db, client))


def test_small_window_shrinks_the_context(retriever):
    """Промпт экзамена занял 4141 токен при окне 4096, и Ollama
    ответила 400: в игровом профиле экзамен просто не работал."""
    from core.prompts import system_prompt

    base = len(system_prompt(Mode.EXAM, legal_context="", rag_enabled=True))
    tight = retriever.context_budget(4096, 320, base)
    roomy = retriever.context_budget(8192, 320, base)
    assert 0 < tight < roomy


def test_budget_is_respected_when_building(retriever):
    class Rule:
        id = 1
        number = "1.1"
        document = "Кодекс"
        text = "Очень длинный текст нормы. " * 60
        sanction = ""

        def full_text(self):
            return self.text

    from core.legal.retriever import Hit

    hits = [Hit(Rule(), 1.0, ["тест"]) for _ in range(5)]
    context = retriever.build_context(hits, max_article_chars=5000,
                                      max_context_chars=900)
    assert len(context) <= 1800, "бюджет выдержек не соблюдается"


def test_no_window_means_no_extra_limit(retriever):
    assert retriever.context_budget(0, 320, 4000) == \
        retriever.max_context_chars


def test_game_profile_window_fits_the_exam_prompt():
    """Регрессия: окно должно вмещать промпт экзамена целиком."""
    cfg = Config.load()
    assert cfg.get("profiles")["game_max"]["llm"]["num_ctx"] >= 6144


def test_startup_unloads_stale_models():
    """Ollama держит модель два часа после обращения, и после смены
    профиля в памяти оставались обе: замер показал 7B и 4B разом —
    восемь гигабайт на карте, где и одной было тесно."""
    source = (Path(__file__).resolve().parent.parent
              / "core" / "orchestrator.py").read_text(encoding="utf-8")
    assert "Выгружаю лишнюю модель" in source
    # Модель эмбеддингов трогать нельзя — без неё встанет поиск.
    assert 'loaded.startswith("bge")' in source


# ------------------------------------------- отказ не тянет чужую статью
def test_refusal_is_recognised():
    from core.orchestrator import _is_refusal

    assert _is_refusal("В найденных статьях ответа нет.")
    assert _is_refusal("В базе нет такой нормы")
    assert not _is_refusal("Штраф до 1.000$. [Дорожный Кодекс, п. 32]")


def test_refusal_gets_no_fallback_article():
    """На экране было «ответа нет» и тут же «скорее всего эта статья»
    с нормой про задержание госслужащих — к вопросу не относящейся."""
    source = (Path(__file__).resolve().parent.parent
              / "core" / "orchestrator.py").read_text(encoding="utf-8")
    assert "if not hits or _is_refusal(answer):" in source


# --------------------------------------------- убийство: кого именно
def test_shooting_alone_is_plain_murder():
    """«Бабка пристрелила воров» — статья про убийство, а не про
    посягательство на жизнь сотрудника."""
    from core.legal.query import normalize

    text = normalize("бабка пристрелила воров огурцов")
    assert "убийство" in text
    assert "посягательство" not in text


def test_shooting_an_officer_is_the_special_article():
    from core.legal.query import normalize

    text = normalize("я убил полицейского при задержании")
    assert "посягательство на жизнь сотрудника" in text


def test_grandmother_is_not_money():
    """«У бабки» — это бабушка, а не «бабки» в смысле денег."""
    from core.legal.query import normalize

    cfg = Config.load()
    text = normalize("у бабки украли ящик огурцов",
                     cfg.get("retrieval.slang"),
                     cfg.get("retrieval.stopwords"),
                     cfg.get("retrieval.combos"))
    assert "денежные средства" not in text
