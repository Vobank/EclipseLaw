"""
Работа над делом в несколько проходов.

Один проход «надиктовка -> документ» упирался в потолок: модель
получала три десятка выдержек вперемешку и выбирала наугад. На живом
деле она поставила статью о телесных повреждениях за незачитанные
права, а половина эпизодов до бумаги не дошла.

Теперь дело собирается как у юриста: разбор на эпизоды, норма под
каждый, черновик, вычитка, правка. Здесь проверяется, что каждый шаг
делает своё дело и что ни один из них не может испортить документ.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.casework import (Episode, episodes_for_prompt,  # noqa: E402
                           filter_review_notes, parse_episodes,
                           revision_is_better, uncovered)
from core.config import Config  # noqa: E402

NL = chr(10)


class Rule:
    def __init__(self, number):
        self.number = number

    def citation(self):
        return f"Процессуальный Кодекс, п. {self.number}"


class Hit:
    def __init__(self, number):
        self.rule = Rule(number)


# ------------------------------------------------------------- разбор
def test_marked_list_is_parsed():
    answer = f"Эпизоды:{NL}- не зачитал права{NL}- обыскал машину без ордера"
    assert parse_episodes(answer) == ["не зачитал права",
                                      "обыскал машину без ордера"]


def test_unmarked_lines_are_parsed():
    """Модель часто отдаёт перечень без дефисов. Требование маркера
    возвращало ноль эпизодов, и весь конвейер откатывался."""
    answer = (f"надел наручники, права не зачитал - 19:40{NL}"
              f"обыскал машину по чужому ордеру - 19:45")
    assert len(parse_episodes(answer)) == 2


def test_headings_are_skipped():
    answer = f"Эпизоды:{NL}Нарушения:{NL}- изъял ноутбук без описи"
    assert parse_episodes(answer) == ["изъял ноутбук без описи"]


def test_short_fragments_are_skipped():
    """По обрывку в три буквы норму не найти."""
    assert parse_episodes(f"- да{NL}- изъял ноутбук без описи") == [
        "изъял ноутбук без описи"]


def test_limit_is_respected():
    answer = NL.join(f"- нарушение номер {i} при задержании" for i in range(30))
    assert len(parse_episodes(answer, limit=12)) == 12


# ------------------------------------------------------ эпизод и норма
def test_uncovered_episodes_are_listed():
    episodes = [Episode("не зачитал права", [Hit("5.1")]),
                Episode("забрал телефон", [])]
    assert uncovered(episodes) == ["забрал телефон"]


def test_prompt_shows_the_missing_norm():
    """Эпизод без нормы должен быть виден: его опишут словами."""
    text = episodes_for_prompt([Episode("забрал телефон", [])])
    assert "норма не найдена" in text


def test_prompt_pairs_episode_with_its_norm():
    text = episodes_for_prompt([Episode("не зачитал права", [Hit("5.1")])])
    assert "не зачитал права" in text and "5.1" in text


# ------------------------------------------------- фильтр замечаний
def test_demand_for_missing_data_is_dropped():
    """По «указать дату подачи» документ дополнят выдумкой."""
    assert filter_review_notes(["Указать дату подачи заявления"], []) == []


def test_note_about_unknown_article_is_dropped():
    notes = ["Добавить ссылку на статью 1.5 Правил подачи"]
    assert filter_review_notes(notes, [Hit("9")]) == []


def test_note_inventing_an_episode_is_dropped():
    """Вычитка знает только рассказ; новых фактов у неё взяться негде."""
    notes = ["Добавить эпизод о том, что ответчик применил силу"]
    assert filter_review_notes(notes, []) == []


def test_useful_notes_survive():
    notes = ["Пропущен эпизод об изъятии ноутбука без описи",
             "Ссылка на статью 5.2 не соответствует эпизоду"]
    assert len(filter_review_notes(notes, [Hit("5.2")])) == 2


# --------------------------------------------------- качество правки
def _document(head="ОПРЕДЕЛЕНИЕ о принятии искового заявления"):
    return f"ОКРУЖНОЙ СУД{NL}{head}{NL}" + "текст дела " * 40


def test_revision_replacing_the_form_is_rejected():
    """Модель меняла «о принятии» на «об отказе в принятии» — это
    противоположный смысл дела."""
    ok, why = revision_is_better(
        _document(), _document("ОПРЕДЕЛЕНИЕ об отказе в принятии"), [], [])
    assert not ok and "вид документа" in why


def test_revision_adding_bad_articles_is_rejected():
    ok, why = revision_is_better(_document(), _document(), ["6.1"],
                                 ["6.1", "12.8"])
    assert not ok and "вне базы" in why


def test_truncated_revision_is_rejected():
    ok, why = revision_is_better(_document(), "ОПРЕДЕЛЕНИЕ", [], [])
    assert not ok and "обрезала" in why


def test_empty_revision_is_rejected():
    assert revision_is_better(_document(), "   ", [], [])[0] is False


def test_good_revision_is_accepted():
    better = _document() + "дополненный раздел"
    assert revision_is_better(_document(), better, ["6.1"], [])[0]


# ------------------------------------------------------------ настройки
def test_pipeline_is_configured():
    cfg = Config.load()
    assert cfg.get("documents.max_episodes") >= 8
    assert cfg.get("documents.articles_per_episode") >= 1
    assert cfg.get("documents.episode_context")


def test_orchestrator_runs_all_passes():
    source = (Path(__file__).resolve().parent.parent
              / "core" / "orchestrator.py").read_text(encoding="utf-8")
    for step in ("_split_episodes", "_review_document", "CASE_REVISE",
                 "revision_is_better", "filter_review_notes"):
        assert step in source, step


def test_review_is_skipped_without_episodes():
    """Без эпизодов вычитке не с чем сверять, и она начинает сочинять."""
    source = (Path(__file__).resolve().parent.parent
              / "core" / "orchestrator.py").read_text(encoding="utf-8")
    assert "Вычитка пропущена: эпизоды не выделены" in source


# ------------------------------------------------- ожидание модели
def test_stream_has_a_hard_deadline():
    """Таймаута requests мало: при стриме Ollama сразу отдаёт
    заголовки, а потом молча грузит модель с диска. Бот так простоял
    пятнадцать минут на «Шаг 1 из 5»."""
    from core.llm.ollama_client import LLMOptions, OllamaClient

    client = OllamaClient("http://127.0.0.1:11434", LLMOptions(model="test"))
    assert client.options.stream_timeout > 0
    source = (Path(__file__).resolve().parent.parent
              / "core" / "llm" / "ollama_client.py").read_text(encoding="utf-8")
    assert "deadline = time.monotonic()" in source


def test_heavy_model_is_preloaded_during_dictation():
    """Загрузка 14B — это десять гигабайт; она должна идти, пока
    человек диктует, а не в момент «составить документ».

    И обязательно мимо очереди конвейера: в ней стоит распознавание
    речи, и прогрев задержал бы показ первой же фразы.
    """
    source = (Path(__file__).resolve().parent.parent
              / "core" / "orchestrator.py").read_text(encoding="utf-8")
    assert "_preload_document_model" in source
    assert "self._worker.submit(self._preload_document_model)" not in source
    assert "threading.Thread(target=self._preload_document_model" in source


def test_ollama_client_is_safe_for_two_threads():
    """Прогрев идёт своим потоком — значит общий пул соединений
    requests делится между потоками и должен быть под замком."""
    source = (Path(__file__).resolve().parent.parent
              / "core" / "llm" / "ollama_client.py").read_text(encoding="utf-8")
    assert "self._lock = threading.RLock()" in source
    assert source.count("with self._lock") >= 3


def test_heavy_model_is_skipped_when_memory_is_short():
    source = (Path(__file__).resolve().parent.parent
              / "core" / "orchestrator.py").read_text(encoding="utf-8")
    assert "document_needs_mb" in source or "_needs_mb" in source
    assert "Памяти мало" in source

def test_worker_failures_are_never_silent():
    """ThreadPoolExecutor прячет исключение внутрь Future. Задача
    падала, поток вставал свободным, в логе после строки об образцах
    было пусто — а на экране навсегда оставалось «Шаг 1 из 5»."""
    source = (Path(__file__).resolve().parent.parent
              / "core" / "orchestrator.py").read_text(encoding="utf-8")
    assert "def _submit(self, func, *args)" in source
    assert 'log.exception("Задача %s упала", name)' in source
    # Мимо обёртки задачи ставить нельзя: одна такая — и сбой снова
    # станет невидимым.
    assert source.count("self._worker.submit(") == 1
    assert "self._worker.submit(guarded)" in source


def test_broken_stream_becomes_a_normal_llm_error():
    """Обрыв связи и таймаут чтения происходят уже после отправки
    запроса — requests бросает их из iter_lines. Наверх обязан уходить
    LLMUnavailable, иначе его никто не ловит."""
    import requests

    from core.llm.ollama_client import (LLMOptions, LLMUnavailable,
                                        Message, OllamaClient)

    client = OllamaClient("http://127.0.0.1:11434", LLMOptions(model="test"))

    class Response:
        status_code = 200

        @staticmethod
        def raise_for_status():
            pass

        @staticmethod
        def close():
            pass

        @staticmethod
        def iter_lines(decode_unicode=False):
            yield '{"message": {"content": "нача"}}'
            raise requests.exceptions.ChunkedEncodingError("связь оборвалась")

    client._session.post = lambda *a, **k: Response()
    with pytest.raises(LLMUnavailable):
        client.chat([Message("user", "дело")])


def test_each_pass_has_its_own_time_budget():
    """Проходов пять: общий потолок означал бы, что один застрявший
    съедает всё ожидание."""
    source = (Path(__file__).resolve().parent.parent
              / "core" / "orchestrator.py").read_text(encoding="utf-8")
    assert 'timeout=self.cfg.get("documents.step_timeout"' in source
