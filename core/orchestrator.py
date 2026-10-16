"""
Связующий слой: хоткей -> запись -> распознавание -> LLM -> оверлей.

Потоки:
  * хук клавиатуры  — свой поток, только кладёт события в очередь;
  * диспетчер       — вызывает обработчики ниже, должен возвращаться быстро;
  * worker          — здесь идут распознавание и генерация, они долгие;
  * UI              — трогается только через сигналы Qt.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from core.answer import (check_answer, dedupe_citations,
                         fold_citations, generate_checked,
                         strip_service_blocks, wrong_attribution)
from core import clipboard
from core.audio.loopback import LoopbackError
from core.audio.mic import MicRecorder, MicRecorderError
from core.audio.other import OtherSpeakerRecorder
from core.documents import (KINDS, TemplateStore, advise,
                            redact_invented_facts, unfilled_fields)
from core.gpu import free_vram_mb
from core.history import HistoryLog
from core.hotkeys import HotkeyEvent, HotkeyManager
from core.legal.checklist import build_checklist
from core.legal.db import LegalDB
from core.legal.importer import ImportError_, LawImporter
from core.legal.parser import ForumParser
from core.legal.query import is_list_question
from core.legal.retriever import (NUMBER_IN_QUERY, NUMBER_WITH_CODE,
                                  Retriever)
from core.legal.vectors import EmbeddingClient, VectorIndex
from core.llm.ollama_client import LLMUnavailable, Message, OllamaClient
from core.prompts import (CASE_EPISODES, CASE_REVIEW,
                          CASE_REVISE, DOCUMENT_RETRY, SUMMARY_PROMPT, document_prompt, system_prompt,
                          verdict_prompt, wants_detail)
from core.session import Mode, SessionManager, Source
from core.stt.whisper_engine import WhisperEngine

log = logging.getLogger(__name__)


# Ответ-отказ: бот не нашёл нормы и говорит об этом прямо.
_REFUSAL = re.compile(
    r"в найденных статьях (?:ответа )?нет|"
    r"в базе (?:этого |такой |такого )?нет|"
    r"не нашл|нет ответа|ответа нет",
    re.IGNORECASE)


def _is_refusal(answer: str) -> bool:
    """Отказ ли это. К отказу пояснять нечего."""
    text = " ".join((answer or "").split())
    return bool(text) and bool(_REFUSAL.search(text))


def _answer_shape(question: str) -> str:
    """Какой формы ответ просит этот вопрос.

    Форма зависит не от режима, а от самого вопроса: «кто относится
    к неприкосновенным лицам» — это перечень, который зачитывают вслух,
    а «что говорит статья 7.1» — просьба объяснить названную статью.
    Один формат на оба случая одинаково плох: в первом он добавляет
    рассуждения к списку, во втором повторяет номер вместо сути.
    """
    if is_list_question(question):
        return "listing"
    if NUMBER_IN_QUERY.search(question) or NUMBER_WITH_CODE.search(question):
        return "article"
    return ""


def _escape(text: str) -> str:
    """Экранирует текст для разметки списка дел.

    Названия дел берутся из надиктованного, а там встречается и «<»,
    и «&» — без экранирования строка молча пропала бы.
    """
    return (text.replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;"))


def _mode_title(mode: str) -> str:
    """Короткое название режима для подписи в списке дел."""
    return {"normal": "обычный", "lawyer": "адвокат", "exam": "экзамен",
            "document": "документ"}.get(mode, mode)


def _same_model(left: str, right: str) -> bool:
    """Одна ли это модель. «qwen2.5:7b» и «qwen2.5:7b-instruct» — разные.

    Ollama хранит имена с тегом, а без тега подразумевается «:latest».
    Сравнивать по части до двоеточия нельзя: у 7B и 14B она общая.
    """
    def full(name: str) -> str:
        return name if ":" in name else f"{name}:latest"

    return full(left) == full(right)


class _PlainHit:
    """Статья без результатов поиска — для сборки контекста заключения.

    build_context и проверка ссылок работают с объектами, у которых
    есть .rule; чек-лист же отдаёт голые статьи.
    """

    def __init__(self, rule):
        self.rule = rule
        self.score = 0.0
        self.sources = ["чек-лист"]

    def why(self) -> str:
        return "чек-лист"

# Единственное место, где описаны действия хоткеев. Отсюда берут текст
# и справка, и меню в трее, и строка подсказки — иначе список в одном
# месте обновляется, а в двух других остаётся старым.
ACTIONS = (
    ("ptt_self", "Задать вопрос голосом", "удерживать"),
    ("ptt_other", "Записать речь собеседника (GTA + Discord)", "удерживать"),
    ("copy_answer", "Скопировать ответ в буфер обмена", ""),
    ("paste_text", "Вставить текст из буфера: вопрос или факты", ""),
    ("full_article", "Показать названные статьи дословно", ""),
    ("expand_answer", "Развернуть последний ответ подробнее", ""),
    ("retry_question", "Переспросить: не расслышал вопрос", ""),
    ("mark_wrong", "Пометить последний ответ как неверный", ""),
    ("cases", "Список дел — открыть мышкой любое прошлое", ""),
    ("search_cases", "Поиск по делам: номер иска или слово", ""),
    ("reset_session", "Сбросить сессию — новая ситуация", ""),
    ("cycle_mode", "Режим: обычный → адвокат → экзамен", ""),
    ("verdict", "Вынести заключение (режим адвоката)", ""),
    ("import_laws", "Импорт правил из буфера обмена", ""),
    ("import_reference", "Импорт справки: услуги, прайсы, бланки", ""),
    ("start_document", "Оформление документа: начать / составить", ""),
    ("toggle_log", "Скрыть/показать лог внизу", ""),
    ("toggle_overlay", "Скрыть/показать оверлей", ""),
    ("help", "Показать этот список", ""),
    ("quit_app", "Выход из приложения", ""),
)


class Orchestrator:
    def __init__(self, cfg, overlay, on_quit=None):
        self.cfg = cfg
        self.overlay = overlay
        # Вызывается при хоткее выхода. main.py передаёт сюда остановку
        # цикла Qt: рабочий поток не имеет права трогать UI напрямую.
        self._on_quit = on_quit

        self.sessions = SessionManager(cfg)
        self.stt = WhisperEngine.from_config(cfg)
        self.llm = OllamaClient.from_config(cfg)

        self.legal_db = LegalDB(cfg.resolve_path("storage.legal_db"))
        self.history = HistoryLog(cfg.resolve_path("storage.history_db"))
        self.importer = LawImporter(self.legal_db, ForumParser.from_config(cfg))
        self.templates = TemplateStore(cfg.resolve_path("storage.templates_dir"))
        # Накопленная надиктовка в режиме документа: реплики не
        # уходят в модель поодиночке, документ собирается из всего
        # сказанного разом.
        self._dictation: list[str] = []

        # Профиль может увести поиск на процессор: рядом с игрой на
        # максималках эти полгигабайта решают, влезет основная модель
        # в видеопамять или драйвер начнёт вытеснять её страницы.
        embeddings_on_cpu = (
            cfg.profile.get("embeddings", {}).get("device") == "cpu")
        self.embeddings = EmbeddingClient(
            cfg.get("llm.host", "http://127.0.0.1:11434"),
            cfg.get("retrieval.embedding_model", "bge-m3"),
            batch_size=cfg.get("retrieval.embedding_batch", 32),
            num_gpu=0 if embeddings_on_cpu else None)
        self.index = VectorIndex(self.legal_db, self.embeddings)
        self.retriever = Retriever.from_config(
            cfg, self.legal_db, self.index)
        self._rag_ready = False
        # Какая модель сейчас в памяти Ollama. Тяжёлые задачи —
        # прокурорское дело, судебный документ, заключение — считает
        # 14B, быстрые вопросы 7B, и вместе они в карту не влезают.
        self._active_model = self.llm.options.model
        # Последний вопрос и найденные по нему статьи: по фразе
        # «расскажи подробнее» искать нечего, нужен прошлый контекст.
        self._last_question = ""
        self._last_hits: list = []
        # Статьи, названные в последнем ответе. Разворачиваются
        # дословно по клавише — без обращения к модели.
        self._last_cited: list = []
        # Последний ответ в том виде, в каком его уносят: без
        # служебных блоков, которые дописывает сама программа.
        self._last_answer = ""
        # Какую бумагу оформляем: вид («court») и название бумаги
        # («ОПРЕДЕЛЕНИЕ о принятии искового заявления»). Пусто —
        # бот угадает вид по надиктовке, как раньше.
        self._document_kind = ""
        self._document_form = ""
        # Последняя запись с микрофона: по хоткею переспроса она
        # пересчитывается более точным декодером, без переговаривания.
        self._last_audio = None
        # Чек-лист процессуальных требований собирается один раз
        # при входе в режим адвоката и живёт до сброса сессии.
        self._checklist = None
        # Роль в режиме адвоката: задержали меня самого или я приехал
        # к задержанному защитником. Вопросы и формулировки разные.
        self._lawyer_role = ""
        # Открыт ли сейчас список дел: по тому же хоткею он
        # закрывается обратно в ленту текущей сессии.
        self._showing_cases = False
        # Строка поиска по делам: что набрано и что сейчас в списке.
        self._search_query = ""
        self._search_hits: list = []
        self._search_active = False

        self.mic = MicRecorder(
            sample_rate=cfg.get("audio.sample_rate", 16000),
            channels=cfg.get("audio.channels", 1),
            device=cfg.get("audio.input_device"),
            tail_padding_ms=cfg.get("ptt.tail_padding_ms", 250),
            max_duration_ms=cfg.get("ptt.max_duration_ms", 30000),
        )
        # Захват собеседника: только процессы из белого списка,
        # системный звук не трогается (ТЗ, раздел 2).
        self.other = OtherSpeakerRecorder(
            cfg.get("audio.loopback_processes", []),
            sample_rate=cfg.get("audio.loopback_sample_rate", 48000),
            channels=cfg.get("audio.loopback_channels", 2),
            max_duration_ms=cfg.get("ptt.max_duration_ms", 30000))
        self._min_ptt_s = cfg.get("ptt.min_duration_ms", 400) / 1000.0
        self._ptt_started_at = 0.0

        # Один рабочий поток: два одновременных запроса к LLM на CPU
        # только замедлят друг друга.
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pipeline")
        self._busy = threading.Lock()
        self._llm_ready = False

        self.hotkeys = HotkeyManager(self._on_hotkey)
        # Клик по делу в списке приходит из потока UI.
        self.overlay.sig_case.connect(self._open_case)
        self.overlay.sig_role.connect(self._choose_lawyer_role)
        self.overlay.sig_form.connect(self._choose_document_form)
        # Окно само сообщает хуку, над каким куском экрана колесо наше.
        self.overlay.set_wheel_zone_sink(self.hotkeys)
        # Вкладка «Дела» в шапке делает то же, что и хоткей.
        self.overlay.sig_cases_tab.connect(self._show_cases)

    # ------------------------------------------------------------------ старт
    def start(self) -> None:
        errors = self.hotkeys.bind_all(
            self.cfg.hotkeys, ptt_actions={"ptt_self", "ptt_other"})
        for err in errors:
            self._log("Система", f"Хоткей не привязан: {err}")
        self.hotkeys.start()

        self.overlay.sig_mode.emit(self.sessions.current.mode.title)
        self._status("idle", self._hotkey_hint())
        self.overlay.sig_answer.emit(self._startup_text())

        self._submit(self._warmup)

    def stop(self) -> None:
        # Размер окна запоминается и при выходе: его могли изменить
        # хоткеем или окно свернули, не отпуская мышь над краем.
        try:
            self.overlay.save_geometry()
        except Exception:
            log.debug("Размер окна не сохранён при выходе", exc_info=True)
        self.hotkeys.stop()
        self.mic.cancel()
        self.other.cancel()
        self._worker.shutdown(wait=False)
        self.legal_db.close()
        self.history.close()

    def _startup_text(self) -> str:
        profile = self.cfg.profile_name
        stt = self.cfg.profile["stt"]
        llm = self.cfg.profile["llm"]
        lines = [
            f"Профиль железа: {profile}",
            f"STT: {stt['model']} на {stt['device']}",
            f"LLM: {llm['model']}",
            "",
        ]
        documents = self.legal_db.documents()
        if documents:
            lines.append("База законов:")
            lines.extend(f"— «{title}»: {count} п."
                         for title, count, _ in documents)
        else:
            lines.append("База законов пуста. Скопируйте страницу правил "
                         "с форума (Ctrl+A, Ctrl+C) и нажмите "
                         f"{self.cfg.hotkeys.get('import_laws')}.")
        lines.extend(["", "Прогрев моделей..."])
        return "\n".join(lines)

    def _warmup(self) -> None:
        """Греем Whisper и проверяем Ollama, чтобы первый PTT не ждал загрузку."""
        ready, reason = self.llm.check_ready()
        self._llm_ready = ready
        self._log("Система", reason)
        if not ready:
            self._status("error", reason)

        try:
            self.stt.load()
            self._log("Система", f"Whisper {self.stt.model_name} загружен")
        except Exception as exc:
            log.exception("Не удалось загрузить Whisper")
            self._status("error", f"Whisper не загрузился: {exc}")
            self.overlay.sig_answer.emit(f"Ошибка загрузки Whisper:\n{exc}")
            return

        self._check_index()

        if self._rag_ready:
            # Матрица векторов грузится с диска почти четыре секунды, и
            # платил за это первый же вопрос. Греем сразу: пользователь
            # в этот момент всё равно читает стартовый экран.
            started = time.monotonic()
            try:
                self.index.search("прогрев", limit=1)
                log.info("Индекс прогрет за %.1f с", time.monotonic() - started)
            except Exception:
                log.exception("Прогрев индекса не удался")

        if ready:
            # И сама модель: без прогрева первый ответ платит за её
            # загрузку почти минуту, а первый вопрос обычно самый
            # нужный — его задают, когда уже задержали.
            self._status("thinking", "Прогреваю модель...")
            self._warm_model()
            self._status("idle", self._hotkey_hint())
            self.overlay.sig_answer.emit(
                "Готов к работе.\n\n"
                f"Задайте вопрос голосом: удерживайте "
                f"{self.cfg.hotkeys.get('ptt_self')}."
            )

    def _check_index(self) -> None:
        """Готов ли смысловой поиск. Без него ассистент отвечать не должен:
        база есть, а сослаться на неё он не сможет."""
        ok, reason = self.embeddings.available()
        if not ok:
            self._rag_ready = False
            self._log("Система", reason)
            return
        stats = self.index.stats()
        self._rag_ready = stats["векторов"] > 0
        if self._rag_ready:
            self._log("Система",
                      f"Поиск по базе готов: {stats['векторов']} векторов")
            if stats["векторов"] < stats["пунктов"]:
                self._log("Система",
                          f"Не проиндексировано пунктов: "
                          f"{stats['пунктов'] - stats['векторов']} — "
                          f"выполните python tools/build_index.py")
        else:
            self._log("Система", "Индекс пуст — python tools/build_index.py")

    def hotkey_help(self) -> str:
        """Полный список сочетаний — для справки и меню в трее."""
        hk = self.cfg.hotkeys
        width = max((len(hk.get(a, "")) for a, _, _ in ACTIONS), default=18)
        lines = ["ГОРЯЧИЕ КЛАВИШИ", ""]
        for action, title, note in ACTIONS:
            combo = hk.get(action)
            if not combo:
                continue
            suffix = f"  ({note})" if note else ""
            lines.append(f"  {combo.ljust(width)}  {title}{suffix}")
        lines.extend([
            "",
            "Колесом мыши над окном листается лента ответов.",
            "Сочетания меняются в config/config.yaml, секция hotkeys.",
            "Ctrl+Alt игра не использует, поэтому конфликтов быть не должно.",
        ])
        return "\n".join(lines)

    def _hotkey_hint(self) -> str:
        """Короткая строка под шапкой: самое частое плюс путь к справке."""
        hk = self.cfg.hotkeys
        return (f"{hk.get('ptt_self')} — вопрос  ·  "
                f"{hk.get('ptt_other')} — собеседник  ·  "
                f"{hk.get('help')} — все клавиши")

    # ------------------------------------------------------------- вспомогательное
    def _status(self, key: str, detail: str = "") -> None:
        self.overlay.sig_status.emit(key, detail)

    def _submit(self, func, *args) -> None:
        """Ставит задачу в очередь так, чтобы её ошибка не потерялась.

        ThreadPoolExecutor прячет исключение внутрь Future, и если его
        никто не спросит, оно исчезает бесследно. На живом запуске это
        выглядело так: человек нажимает «составить документ», на экране
        появляется «Шаг 1 из 5» — и остаётся там навсегда, а в логе
        после строки об образцах пусто. Поток при этом не висел: задача
        уже упала, и стек показал его свободным.

        Молчаливый сбой хуже любого сбоя: непонятно даже, ждать или
        нажимать заново. Поэтому ошибку пишем в лог со стеком и
        показываем человеку прямо в окне.
        """
        name = getattr(func, "__name__", str(func))

        def guarded() -> None:
            try:
                func(*args)
            except Exception as exc:
                log.exception("Задача %s упала", name)
                self._log("Ошибка", f"{name}: {exc}")
                self.overlay.sig_answer_chunk.emit(
                    f"\n\n[!] Сбой: {exc}\n"
                    "Подробности в logs/app.log. Можно нажать заново.\n")
                self._status("idle", "Сбой — подробности в логе")

        self._worker.submit(guarded)

    def _log(self, who: str, text: str) -> None:
        # Дублируем в файл. Раньше эти сообщения жили только в окошке на
        # экране, и когда бот ответил «в базе не нашлось», разобраться
        # постфактум было не по чему: ни готовности поиска, ни числа
        # найденных статей в логе не оставалось.
        log.info("[%s] %s", who, text)
        self.overlay.sig_log.emit(who, text)

    # ----------------------------------------------------------------- хоткеи
    def _on_hotkey(self, event: HotkeyEvent) -> None:
        action, kind = event.action, event.kind
        log.debug("Хоткей: %s / %s", action, kind)

        if action == "ptt_self":
            if kind == "pressed":
                self._close_search()
                self._start_recording()
            else:
                self._finish_recording()
            return

        if action == "ptt_other":
            if kind == "pressed":
                self._start_other()
            else:
                self._finish_other()
            return

        if kind != "pressed":
            return

        if action == "reset_session":
            self._reset_session()
        elif action == "cycle_mode":
            self._cycle_mode()
        elif action == "toggle_overlay":
            # Спрятали окно — поиск закрываем принудительно: иначе хук
            # продолжит глотать буквы, а спросить их будет негде, и в
            # игре перестанет работать клавиатура.
            self._close_search()
            self.overlay.toggle_visibility()
        elif action == "verdict":
            self._request_verdict()
        elif action == "toggle_log":
            self.overlay.toggle_log()
        elif action == "start_document":
            self._toggle_document_mode()
        elif action == "help":
            self.overlay.sig_answer.emit(self.hotkey_help())
            self._status("idle", self._hotkey_hint())
        elif action == "quit_app":
            log.info("Выход по хоткею")
            self._log("Система", "Завершение работы")
            if self._on_quit is not None:
                self._on_quit()
        elif action == "import_reference":
            self.request_import(reference=True)
        elif action == "import_laws":
            # Импорт читает буфер и пишет в базу — в потоке диспетчера
            # такое держать нельзя, он обязан освободиться сразу.
            self.request_import()
        elif action == "expand_answer":
            self._expand_last()
        elif action == "full_article":
            self._show_full_article()
        elif action == "copy_answer":
            self._copy_answer()
        elif action == "paste_text":
            # Чтение буфера и ответ — в рабочем потоке: поток
            # диспетчера обязан освободиться сразу.
            self._submit(self._paste_text)
        elif action == "retry_question":
            self._submit(self._retry_question)
        elif action == "mark_wrong":
            self._mark_wrong()
        elif action == "cases":
            self._show_cases()
        elif action == "search_cases":
            self._toggle_search()
        elif action == "search_edit":
            self._search_type(event.text)
        elif action == "search_back":
            self._search_type("", back=True)
        elif action == "search_close":
            self._close_search()
        elif action == "search_open":
            self._open_first_found()
        elif action in ("scroll_up", "scroll_down"):
            # Колесо перехвачено хуком: окно фокуса не имеет и само
            # такого события не увидит.
            self.overlay.sig_scroll.emit(1 if action == "scroll_up" else -1)
        else:
            log.warning("Неизвестное действие хоткея: %s", action)

    def _checklist_hits(self) -> list:
        """Статьи чек-листа в виде, понятном проверке ссылок."""
        if self._checklist is None:
            return []
        return [_PlainHit(a) for item in self._checklist.grounded
                for a in item.articles]

    # ------------------------------------------------------- режим адвоката
    def _prepare_checklist(self) -> None:
        """Собирает чек-лист процессуальных требований из базы.

        Делается один раз при входе в режим, а не на каждый вопрос:
        девять поисков по базе — это девять обращений к модели
        эмбеддингов, и держать их в диалоге незачем.
        """
        if not self._rag_ready:
            self.overlay.sig_answer.emit(
                "Режим адвоката включён, но поиск по базе не готов —\n"
                "чек-лист собрать не из чего.\n\n"
                "Выполните: python tools/build_index.py")
            return

        if not self._lawyer_role:
            # Роль ещё не выбрана в этой сессии — спрашиваем. Набор
            # вопросов от этого зависит целиком: у защитника, который
            # только приехал, первый вопрос не «за что меня задержали»,
            # а «допустите меня к подзащитному».
            self._ask_lawyer_role()
            return

        role = self._role_config(self._lawyer_role)
        self._status("thinking",
                     f"Собираю вопросы: {role.get('title', self._lawyer_role)}...")
        try:
            self._checklist = build_checklist(
                self.retriever,
                topics=role.get("checklist"),
                per_topic=self.cfg.get("lawyer.articles_per_topic", 2))
        except Exception as exc:
            log.exception("Не удалось собрать чек-лист")
            self._status("error", f"Чек-лист не собран: {exc}")
            return

        grounded = self._checklist.grounded
        self._show_checklist()
        self._log("Система", f"Чек-лист: {len(grounded)} пунктов с опорой в базе")
        self._status("ready", self._hotkey_hint())

    def _role_config(self, role: str) -> dict:
        """Настройки роли адвоката. Старый конфиг без ролей тоже годится."""
        roles = self.cfg.get("lawyer.roles") or {}
        found = roles.get(role)
        if found:
            return found
        # Совместимость с конфигом, где чек-лист был один и лежал
        # прямо в lawyer.checklist.
        return {"title": "Задержали меня",
                "checklist": self.cfg.get("lawyer.checklist")}

    def _ask_lawyer_role(self) -> None:
        """Спрашивает, чья это ситуация: моя или клиента."""
        roles = self.cfg.get("lawyer.roles") or {}
        if not roles:
            self._choose_lawyer_role("detained")
            return

        rows = ["<div style='font-size:15px; font-weight:600; "
                "margin-bottom:10px'>РЕЖИМ АДВОКАТА</div>",
                "<div style='color:#d1d5db; margin-bottom:12px'>"
                "Кого защищаем?</div>"]
        for key, role in roles.items():
            rows.append(
                f"<div style='margin-bottom:12px'>"
                f"<a href='role:{key}' style='color:#93c5fd; "
                f"text-decoration:none; font-size:15px'>"
                f"{_escape(role.get('title', key))}</a>"
                f"<br><span style='color:#9ca3af; font-size:11px'>"
                f"{_escape(role.get('hint', ''))}</span></div>")
        rows.append("<div style='color:#9ca3af; font-size:11px; "
                    "margin-top:10px'>Щёлкните мышкой. Сменить роль — "
                    "выйти из режима и вернуться.</div>")
        self.overlay.show_html("".join(rows))
        self._status("idle", "Выберите роль")

    def _choose_lawyer_role(self, role: str) -> None:
        """Роль выбрана — собираем её чек-лист."""
        if self.sessions.current.mode is not Mode.LAWYER:
            return
        self._lawyer_role = role or "detained"
        self.overlay.show_transcript()
        self._submit(self._prepare_checklist)

    def _show_checklist(self) -> None:
        """Готовые вопросы к сотруднику — то, что произносится вслух.

        Формулировки заготовлены и лежат в конфиге, а не сочиняются
        моделью: одиннадцать вопросов на CPU-профиле считались бы минут
        десять, а нужны они в первую же секунду задержания.
        """
        if self._checklist is None:
            return
        hk = self.cfg.hotkeys
        grounded = self._checklist.grounded
        done = sum(1 for item in grounded if item.covered)

        lines = [
            "РЕЖИМ АДВОКАТА — Вова Хэтфилд",
            f"Пройдено {done} из {len(grounded)} процессуальных требований",
            "",
        ]
        current = self._checklist.next_question()
        if current is not None:
            lines.extend([
                "СЕЙЧАС СПРОСИТЬ:",
                f"  «{current.spoken_question()}»",
            ])
            if current.expected:
                # Что должно прозвучать в ответ, если процедура
                # соблюдена. Иначе ответ сотрудника нечем мерить:
                # нарушение звучит так же уверенно, как соблюдение.
                lines.append(f"  ждём в ответ: {current.expected}")
            lines.extend([
                "",
                "─" * 34,
                "",
            ])
        lines.append(self._checklist.as_overlay_text())
        lines.extend([
            f"{hk.get('ptt_self')} — спросить своё   ·   "
            f"{hk.get('ptt_other')} — ответ сотрудника",
            f"{hk.get('verdict')} — вынести заключение",
        ])
        self.overlay.sig_answer.emit("\n".join(lines))

    def _request_verdict(self) -> None:
        if self.sessions.current.mode is not Mode.LAWYER:
            self._status("idle", "Заключение выносится только в режиме адвоката")
            return
        self._submit(self._make_verdict)

    def _make_verdict(self) -> None:
        """Итоговое заключение по задержанию (ТЗ 5.4)."""
        session = self.sessions.current
        if not session.exchanges:
            self.overlay.sig_answer.emit(
                "Заключение выносить не по чему: в этой сессии не было "
                "ни одного вопроса.")
            return

        with self._busy:
            self._status("thinking", "Готовлю заключение по задержанию...")

            # В заключение идут все статьи, всплывшие за сессию, плюс
            # чек-лист: нарушение процедуры обычно касается пункта,
            # о котором сотрудника даже не спросили.
            hits = list(self._last_hits)
            seen = {h.rule.id for h in hits}
            if self._checklist is not None:
                for item in self._checklist.grounded:
                    for article in item.articles:
                        if article.id not in seen:
                            seen.add(article.id)
                            hits.append(_PlainHit(article))

            context = self.retriever.build_context(hits)
            transcript = "\n".join(
                f"{e.source.title}: {e.question}\nАдвокат: {e.answer}"
                for e in session.exchanges)

            messages = [
                Message("system", verdict_prompt(context)),
                Message("user", "Стенограмма задержания:\n\n" + transcript),
            ]

            self.overlay.begin_answer()
            pieces: list[str] = []
            try:
                for chunk in self.llm.stream_chat(
                        messages, model=self._model_for("verdict")):
                    pieces.append(chunk)
                    self.overlay.sig_answer_chunk.emit(chunk)
            except LLMUnavailable as exc:
                log.error("Заключение не составлено: %s", exc)
                self._status("error", str(exc))
                return

            verdict = "".join(pieces).strip()
            # Тем же набором проверок, что и обычный ответ: в
            # заключении для суда выдуманный срок опаснее всего.
            invented = check_answer(self.retriever, verdict, hits)
            if invented:
                warning = ("\n\n[!] ПРОВЕРЬТЕ ВРУЧНУЮ: в найденных статьях "
                           "нет " + ", ".join(invented) +
                           " — в заключении для суда это критично.")
                self.overlay.sig_answer_chunk.emit(warning)
                verdict += warning

            # Промаха с самим составом проверка номеров не ловит:
            # номер настоящий, статья существует, но привлекать по ней
            # нельзя. Это отдельная проверка.
            misattributed = wrong_attribution(self.retriever, verdict, hits)
            if misattributed:
                self.overlay.sig_answer_chunk.emit(misattributed)
                verdict += misattributed

            spoken_verdict = verdict
            self._last_answer = spoken_verdict
            verdict += self._explain_cited(verdict, hits)

            session.add(Source.SYSTEM, "Вынести заключение", spoken_verdict)
            self.history.record(
                session_id=session.id, mode=session.mode.value,
                source=Source.SYSTEM.title,
                question="Вынести заключение", answer=verdict,
                articles=self.retriever.articles_mentioned(hits),
                invented=invented)
            self._log("Система", "Заключение составлено")
            self._status("ready", self._hotkey_hint())

    def _expand_last(self) -> None:
        """Хоткей «развернуть»: то же, что сказать «расскажи подробнее»."""
        if not self._last_question:
            self._status("idle", "Сначала задайте вопрос")
            return
        self._submit(
            self._answer, f"{self._last_question} — расскажи подробнее",
            Source.SELF)

    def _reset_session(self) -> None:
        session = self.sessions.reset()
        self._last_question, self._last_hits, self._last_cited = "", [], []
        self._lawyer_role = ""
        if self.sessions.current.mode is Mode.LAWYER:
            self._submit(self._prepare_checklist)
        else:
            self._checklist = None
        self.overlay.clear_log()
        # Лента вопросов живёт до сброса — здесь её и чистим.
        self.overlay.clear_answers()
        self._log("Система", f"Новая сессия {session.id}")
        self.overlay.sig_answer.emit("Контекст сброшен. Новая ситуация.")
        self._status("idle", self._hotkey_hint())

    def _cycle_mode(self) -> None:
        mode = self.sessions.cycle_mode()
        self.overlay.sig_mode.emit(mode.title)
        self.overlay.clear_log()
        self._log("Система", f"Режим: {mode.title}")

        if mode is Mode.LAWYER:
            # Чек-лист собирается именно здесь. Раньше он строился
            # только при сбросе сессии, и режим адвоката, включённый
            # хоткеем посреди задержания, оставался без готовых
            # вопросов — то есть без главного, ради чего он нужен.
            self._lawyer_role = ""
            self._submit(self._prepare_checklist)
            return

        self._checklist = None
        self._lawyer_role = ""
        if mode is Mode.EXAM:
            self.overlay.sig_answer.emit(
                "РЕЖИМ ЭКЗАМЕНА\n\n"
                "Ответ — связное объяснение своими словами на два-четыре "
                "предложения со ссылкой на статью: его можно произнести "
                "или набрать как есть.\n\n"
                "Статей в поиск идёт больше, длинные не обрезаются — "
                "перечисления доходят целиком. Названная статья, которой "
                "нет в базе, ловится проверкой, и ответ переписывается "
                "заново.\n\n"
                f"Свой вопрос — {self.cfg.hotkeys.get('ptt_self')}.\n"
                f"Вопрос экзаменатора — {self.cfg.hotkeys.get('ptt_other')}: "
                "держите клавишу, пока он говорит, и повторять за ним "
                "не придётся.")
            # Прогрев: первый вопрос иначе платит за загрузку модели
            # сорок секунд, а на экзамене это половина ответа.
            self._submit(self._warm_model)
            return

        self.overlay.sig_answer.emit("Обычный режим. Задайте вопрос голосом.")

    def _warm_model(self) -> None:
        """Прогоняет через модель один токен, чтобы она была в памяти.

        Перед этим выгружает чужие модели: Ollama держит их два часа
        после последнего обращения, и после смены профиля в памяти
        оставалась и старая. Замер показал 7B и 4B одновременно —
        восемь гигабайт на карте, где и одной было тесно.
        """
        wanted = self.cfg.profile["llm"]["model"]
        try:
            for loaded in self.llm.loaded_models():
                # Модель эмбеддингов не трогаем: без неё встанет поиск.
                if loaded.startswith("bge") or _same_model(loaded, wanted):
                    continue
                log.info("Выгружаю лишнюю модель: %s", loaded)
                self.llm.unload(loaded)
        except Exception:
            log.debug("Не удалось осмотреть загруженные модели",
                      exc_info=True)

        try:
            self.llm.chat([Message("user", "готов")], num_predict=1)
            log.info("Модель прогрета")
        except LLMUnavailable as exc:
            log.warning("Прогрев не удался: %s", exc)

    # ---------------------------------------------------------------- импорт
    def request_import(self, reference: bool = False) -> None:
        """Запуск импорта: из хоткея или из меню значка в трее."""
        if self._busy.locked():
            self._status("importing", "Импорт начнётся после текущего ответа")
        self._submit(self._import_laws, reference)

    def _import_laws(self, reference: bool = False) -> None:
        """Ctrl+A / Ctrl+C на странице форума, затем этот хоткей."""
        kind = "справочной страницы" if reference else "правил"
        self._status("importing", f"Импорт {kind}: читаю буфер...")
        self.overlay.sig_answer.emit(f"Импорт {kind}: читаю буфер обмена...")
        started = time.monotonic()
        try:
            result = (self.importer.import_reference_from_clipboard()
                      if reference else
                      self.importer.import_from_clipboard())
        except ImportError_ as exc:
            log.warning("Импорт не выполнен: %s", exc)
            self._status("error", "Импорт не выполнен")
            self.overlay.sig_answer.emit(f"ИМПОРТ НЕ ВЫПОЛНЕН\n\n{exc}")
            return
        except Exception as exc:
            log.exception("Сбой импорта")
            self._status("error", f"Сбой импорта: {exc}")
            self.overlay.sig_answer.emit(f"ИМПОРТ НЕ ВЫПОЛНЕН\n\nСбой: {exc}")
            return

        elapsed = time.monotonic() - started
        lines = [
            f"ГОТОВО — «{result.document}»",
            f"обработано за {elapsed:.1f} с",
            "",
            f"Пунктов всего  : {result.total}",
            f"  новых        : {result.added}",
            f"  обновлено    : {result.updated}",
            f"  без изменений: {result.unchanged}",
        ]
        if result.warnings:
            lines.append("")
            lines.append(f"Предупреждения ({len(result.warnings)}):")
            lines.extend(f"— {w}" for w in result.warnings[:5])

        documents = self.legal_db.documents()
        lines.append("")
        lines.append("Сейчас в базе:")
        lines.extend(f"— «{title}»: {count} п."
                     for title, count, _ in documents)

        self.overlay.sig_answer.emit("\n".join(lines))
        self._log("Система", result.summary())
        # Отдельный статус вместо общего «Готово»: пользователю нужно
        # видеть именно то, что закончился импорт, а не ответ на вопрос.
        self._status("imported",
                     f"«{result.document}»: {result.total} п. "
                     f"(новых {result.added}, обновлено {result.updated})")

        # Считаем векторы сразу же. Раньше импорт их не трогал, и
        # свежие пункты не находились смысловым поиском — молча, без
        # единой ошибки: бот отвечал «в базе не нашлось», хотя страница
        # только что импортировалась. Узнать об этом можно было лишь
        # запустив tools/build_index.py вручную, о чём легко забыть.
        self._index_new_rules()

    def _index_new_rules(self) -> None:
        """Достраивает векторы для пунктов, у которых их ещё нет."""
        stats = self.index.stats()
        missing = stats["пунктов"] - stats["векторов"]
        if missing <= 0:
            return

        self._status("importing", f"Индексирую {missing} новых пунктов...")
        self._log("Система", f"Индексация: {missing} пунктов без вектора")
        try:
            done = self.index.build(
                on_progress=lambda ready, total: self._status(
                    "importing", f"Индексация: {ready} из {total}"))
        except Exception as exc:
            log.exception("Индексация не удалась")
            self._status("error", f"Индексация не удалась: {exc}")
            self.overlay.sig_answer_chunk.emit(
                "\n\n[!] ВЕКТОРЫ НЕ ПОСЧИТАНЫ — новые пункты не будут "
                "находиться смысловым поиском.\nВыполните: "
                "python tools/build_index.py")
            return

        self._rag_ready = True
        self._log("Система", f"Индексация завершена: {done.summary()}")
        self._status("imported", "Готово — новые пункты в поиске. "
                     + self._hotkey_hint())
        self.overlay.sig_answer_chunk.emit(
            f"\n\nПроиндексировано новых пунктов: {missing}. "
            "Они уже участвуют в поиске.")

    # ------------------------------------------------------------------- PTT
    def _start_recording(self) -> None:
        if self._busy.locked():
            self._status("thinking", "Дождитесь ответа на предыдущий вопрос")
            return
        try:
            self.mic.start()
        except MicRecorderError as exc:
            log.error("%s", exc)
            self._status("error", str(exc))
            return
        self._ptt_started_at = time.monotonic()
        self._status("recording", "Говорите, отпустите клавишу по окончании")

    def _finish_recording(self) -> None:
        if not self.mic.is_recording:
            return
        held = time.monotonic() - self._ptt_started_at
        if held < self._min_ptt_s:
            self.mic.cancel()
            self._status("idle", f"Слишком короткое нажатие ({held:.1f} с)")
            return

        audio = self.mic.stop()
        if audio.size == 0:
            self._status("error", "Микрофон не отдал звук")
            return
        self._status("transcribing", "Распознавание речи...")
        self._submit(self._run_pipeline, audio, Source.SELF)

    # ------------------------------------------------ режим документов
    def _toggle_document_mode(self) -> None:
        """Первое нажатие — начать надиктовку, второе — составить документ."""
        if self.sessions.current.mode is Mode.DOCUMENT:
            if not self._dictation:
                self.sessions.set_mode(Mode.NORMAL)
                self.overlay.sig_mode.emit(Mode.NORMAL.title)
                self.overlay.sig_answer.emit("Оформление документа отменено.")
                return
            self._submit(self._make_document)
            return

        self.sessions.set_mode(Mode.DOCUMENT)
        self._dictation = []
        self._document_form = ""
        self.overlay.sig_mode.emit(Mode.DOCUMENT.title)
        # Какую именно бумагу писать, бот по надиктовке не угадает:
        # определений в образцах почти три десятка, и все разные.
        # Спрашиваем сразу, до надиктовки.
        self._ask_document_form()
        return

        counts = self.templates.counts()
        lines = ["ОФОРМЛЕНИЕ ДОКУМЕНТА", ""]
        for kind, title in KINDS.items():
            lines.append(f"  {title}: образцов {counts.get(kind, 0)}")
        lines.extend([
            "",
            f"Надиктуйте факты: {self.cfg.hotkeys.get('ptt_self')} "
            "(можно несколько раз).",
            f"Готово — нажмите {self.cfg.hotkeys.get('start_document')} "
            "ещё раз, документ будет составлен.",
            "",
            "Вид документа определяется по надиктовке; если нужно "
            "судопроизводство — скажите это прямо.",
        ])
        if not counts.get("proceedings"):
            lines.extend([
                "",
                "Образцов пока нет. Добавить:",
                "  python tools/add_template.py --clipboard",
            ])
        # Сразу показываем, что понадобится назвать: искать это уже
        # посреди надиктовки поздно.
        lines.extend(["", "─" * 34, "", advise("").as_text()])
        self.overlay.sig_answer.emit("\n".join(lines))
        self._status("idle", "Режим документа: надиктуйте факты")

    def _ask_document_form(self) -> None:
        """Спрашивает, какую бумагу оформляем.

        По надиктовке бот угадывает только тему. Судейских образцов
        почти три десятка — определение о принятии иска, о переносе
        заседания, приговор, решение, — и выбрать между ними может
        только человек.
        """
        rows = ["<div style='font-size:15px; font-weight:600; "
                "margin-bottom:8px'>ЧТО ОФОРМЛЯЕМ?</div>"]
        for kind, title in KINDS.items():
            rows.append(
                f"<div style='margin-bottom:8px'>"
                f"<a href='form:{kind}|' style='color:#93c5fd; "
                f"text-decoration:none; font-size:15px'>{_escape(title)}"
                f"</a></div>")

        for kind in KINDS:
            forms = self.templates.forms(kind)
            if not forms:
                continue
            rows.append("<div style='color:#9ca3af; font-size:11px; "
                        "margin:10px 0 6px'>по образцу:</div>")
            for name, count in forms:
                rows.append(
                    f"<div style='margin-bottom:5px'>"
                    f"<a href='form:{kind}|{_escape(name)}' "
                    f"style='color:#93c5fd; text-decoration:none'>"
                    f"{_escape(name)}</a>"
                    f"<span style='color:#6b7280; font-size:11px'> "
                    f"&middot; образцов {count}</span></div>")

        rows.append("<div style='color:#9ca3af; font-size:11px; "
                    "margin-top:10px'>Щёлкните мышкой. Можно выбрать "
                    "вид сверху — тогда бумагу бот подберёт сам.</div>")
        self.overlay.show_html("".join(rows))
        self._status("idle", "Выберите, что оформляем")

    def _preload_document_model(self) -> None:
        """Греет тяжёлую модель, пока человек диктует.

        Загрузка 14B с диска — это десять гигабайт и до минуты
        ожидания. Раньше она начиналась в момент «составить документ»,
        и человек смотрел на «Шаг 1 из 5», не понимая, жив ли бот.
        Теперь она идёт фоном, пока идёт надиктовка.

        Отдельным потоком, а не через очередь конвейера: в очереди
        стоит распознавание речи, и прогрев задержал бы показ первой
        же надиктованной фразы на всё время загрузки.
        """
        try:
            model = self._model_for("document")
            self.llm.chat([Message("user", "готов")], num_predict=1,
                          model=model)
            log.info("Модель для документов прогрета: %s", model)
        except LLMUnavailable as exc:
            log.warning("Прогрев модели документов не удался: %s", exc)
        except Exception:
            log.exception("Сбой прогрева модели документов")

    def _choose_document_form(self, value: str) -> None:
        """Вид выбран — переходим к надиктовке."""
        if self.sessions.current.mode is not Mode.DOCUMENT:
            return
        kind, _, form = value.partition("|")
        self._document_kind = kind if kind in KINDS else ""
        self._document_form = form.strip()
        self.overlay.show_transcript()

        hk = self.cfg.hotkeys
        title = self._document_form or KINDS.get(self._document_kind, "документ")
        self.overlay.sig_answer.emit(
            f"ОФОРМЛЕНИЕ: {title}\n\n"
            f"Надиктуйте факты: {hk.get('ptt_self')} "
            f"(или вставьте текст: {hk.get('paste_text')}).\n"
            f"Готово — {hk.get('start_document')} ещё раз.")
        self._status("idle", "Надиктуйте факты")
        # Пока человек диктует, тяжёлая модель успеет загрузиться.
        threading.Thread(target=self._preload_document_model,
                         name="preload", daemon=True).start()

    def _show_dictation(self) -> None:
        """Показывает надиктованное и подсказывает, чего не хватает.

        Подсказка считается признаками, а не моделью: она нужна сразу
        после каждой реплики, а обращение к модели на CPU-профиле стоит
        минуты — такой подсказкой просто перестали бы пользоваться.
        """
        dictation = "\n".join(self._dictation)
        words = len(dictation.split())
        advice = advise(dictation, self._pick_kind(dictation))

        hk = self.cfg.hotkeys
        lines = [advice.as_text(words), "", "─" * 34, "", "СКАЗАНО:"]
        lines.extend(f"  {i}. {text}"
                     for i, text in enumerate(self._dictation, 1))
        lines.extend([
            "",
            f"{hk.get('ptt_self')} — дополнить   ·   "
            f"{hk.get('start_document')} — составить документ",
        ])
        self.overlay.sig_answer.emit("\n".join(lines))

        if advice.ready:
            self._status("ready", "Можно составлять документ")
        else:
            self._status("idle",
                         f"Осталось назвать: {advice.missing[0].title}")

    def _split_episodes(self, dictation: str) -> list:
        """Шаг 1-2: разбираем дело на эпизоды и ищем норму под каждый.

        Поиск по надиктовке целиком приносит общее «про задержание»,
        а поиск по эпизоду «обыскал машину без ордера» — статью об
        ордерах. Разница решающая: половина эпизодов иначе остаётся
        без нормы, а модель подставляет к ним что попало.
        """
        self._status("thinking", "Разбираю дело на эпизоды...")
        self.overlay.sig_answer_chunk.emit(
            "Шаг 1 из 5: разбираю дело на эпизоды...\n")
        started = time.monotonic()
        try:
            answer = self.llm.chat(
                [Message("system", CASE_EPISODES),
                 Message("user", dictation)],
                num_predict=self.cfg.get("documents.episodes_predict", 500),
                model=self._model_for("document"),
                timeout=self.cfg.get("documents.step_timeout", 120))
        except LLMUnavailable as exc:
            # Не беда: дальше дело пойдёт одним куском, как раньше.
            # Хуже было бы молча ждать — и человек ждал.
            log.warning("Разбор на эпизоды не удался: %s", exc)
            self.overlay.sig_answer_chunk.emit(
                "Разбор на эпизоды не вышел — пишу дело одним куском." "\n")
            return []

        texts = parse_episodes(
            answer, limit=self.cfg.get("documents.max_episodes", 12))
        if not texts:
            log.info("Эпизоды не выделены, дело пойдёт одним куском")
            return []

        episodes = []
        per_episode = self.cfg.get("documents.articles_per_episode", 2)
        for index, text in enumerate(texts, 1):
            self._status("thinking",
                         f"Ищу норму под эпизод {index} из {len(texts)}...")
            try:
                # Эпизод ищем с правовым контекстом: «изъял ноутбук
                # без описи» сам по себе приводит к уставу LSPD про
                # амуницию, а «сотрудник при задержании: изъял...» —
                # к статье 4.8.5 Процессуального кодекса об изъятии.
                hits = self.retriever.retrieve(
                    f"{self.cfg.get('documents.episode_context', '')}{text}",
                    limit=per_episode)
            except Exception:
                log.exception("Поиск нормы под эпизод не отработал")
                hits = []
            episodes.append(Episode(text=text, hits=hits))

        covered = sum(1 for e in episodes if e.covered)
        self._log("База", f"эпизодов {len(episodes)}, с нормой {covered}")
        self.overlay.sig_answer_chunk.emit(
            f"Шаг 2 из 5: эпизодов {len(episodes)}, норма найдена "
            f"у {covered}. ({time.monotonic() - started:.0f} с)\n")
        log.info("Эпизоды дела: %s",
                 "; ".join(f"{e.text[:40]}={len(e.hits)}" for e in episodes))
        return episodes

    def _review_document(self, document: str, episodes: list,
                         law_context: str, law_hits: list) -> list[str]:
        """Шаг 4: вычитка черновика чужими глазами.

        Отдельный проход, потому что за одно действие модель либо
        пишет, либо проверяет. Совмещать не выходит: в том же ответе
        она уверяет, что всё хорошо.
        """
        if not self.cfg.get("documents.review", True):
            return []
        if not episodes:
            # Без эпизодов вычитке не с чем сверять документ, и она
            # начинает сочинять: предлагала «добавить эпизод о том,
            # что ответчик применил силу», хотя об этом никто не
            # говорил. Лучше вовсе не вычитывать.
            log.info("Вычитка пропущена: эпизоды не выделены")
            return []
        self._status("thinking", "Вычитываю черновик...")
        self.overlay.sig_answer_chunk.emit(
            "\n\nШаг 4 из 5: вычитываю черновик...\n")
        material = (f"ЭПИЗОДЫ И НОРМЫ:\n{episodes_for_prompt(episodes)}\n\n"
                    f"НОРМЫ ЦЕЛИКОМ:\n{law_context[:4000]}\n\n"
                    f"ДОКУМЕНТ:\n{document}")
        try:
            answer = self.llm.chat(
                [Message("system", CASE_REVIEW), Message("user", material)],
                num_predict=self.cfg.get("documents.review_predict", 600),
                model=self._model_for("document"))
        except LLMUnavailable as exc:
            log.warning("Вычитка не удалась: %s", exc)
            return []

        if "ЗАМЕЧАНИЙ НЕТ" in answer.upper():
            self._log("Проверка", "вычитка: замечаний нет")
            return []
        notes = parse_episodes(answer, limit=10)
        # Вычитка охотно требует дописать дату подачи и сослаться на
        # норму, которой в базе нет. По таким замечаниям документ
        # дополняется выдумкой — правка приносит вред, а не пользу.
        all_hits = list(law_hits) + [h for e in episodes for h in e.hits]
        notes = filter_review_notes(notes, all_hits)
        if notes:
            self._log("Проверка", f"вычитка нашла замечаний: {len(notes)}")
            log.info("Замечания вычитки: %s", "; ".join(n[:60] for n in notes))
        return notes

    def _document_law(self, dictation: str, form: str = "") -> list:
        """Статьи, на которые опирается документ.

        Ищем по самой надиктовке: в ней и названы обстоятельства
        («не зачитал права», «обыскал без ордера») и номера статей,
        которые человек помнит. Берём с запасом — в документе важнее
        полнота, чем скорость: лучше подать модели лишнюю норму, чем
        заставить её вспоминать номер по памяти.
        """
        if not self._rag_ready:
            return []
        limit = self.cfg.get("documents.articles", 14)
        found: list = []
        seen: set = set()
        try:
            for hit in self.retriever.retrieve(dictation, limit=limit):
                if hit.rule.id not in seen:
                    seen.add(hit.rule.id)
                    found.append(hit)
            # Отдельно — по названию бумаги: «определение о принятии
            # искового заявления» тянет за собой нормы о подсудности
            # и о сроках обжалования, которых в надиктовке нет.
            queries = [form] if form else []
            # И обязательный минимум для судебных бумаг. Без него
            # проверка ругалась на «36 часов» — срок обжалования,
            # который стоит в каждом определении и в законе есть,
            # просто в выдержки по делу не попадал.
            queries.extend(self.cfg.get("documents.always_search") or [])
            for query in queries:
                if not query:
                    continue
                for hit in self.retriever.retrieve(query, limit=3):
                    if hit.rule.id not in seen:
                        seen.add(hit.rule.id)
                        found.append(hit)
        except Exception:
            log.exception("Поиск статей для документа не отработал")
        return found

    # Кодексы, по которым вообще привлекают к ответственности. Устав
    # или процессуальная норма говорят, как положено действовать,
    # а наказание устанавливают только эти.
    _LIABILITY_DOCS = ("уголовный", "административный", "этический")

    def _charges(self, dictation: str) -> list:
        """Статьи обвинения — подбираем сами, а не просим модель выбрать.

        Проверено на 14B: из двух десятков выдержек она уверенно
        выбирала не то. За незачитанные права ставила УК 6.1 —
        нанесение телесных повреждений, за отсутствие видеозаписи —
        УК 12.8, незаконный оборот оружия. Номера настоящие, статьи
        не те, и в готовом деле это заметит только юрист.

        Поэтому выбор делаем поиском по базе: ищем нормы об
        ответственности должностного лица, а модели подаём готовый
        короткий список.
        """
        if not self._rag_ready:
            return []
        found: list = []
        seen: set = set()
        queries = self.cfg.get("documents.charge_search") or []
        for query in queries:
            try:
                hits = self.retriever.retrieve(f"{query} {dictation}"[:400],
                                               limit=4)
            except Exception:
                log.exception("Подбор статей обвинения не отработал")
                continue
            for hit in hits:
                document = (hit.rule.document or "").lower()
                if not any(kind in document for kind in self._LIABILITY_DOCS):
                    continue
                if hit.rule.id in seen:
                    continue
                seen.add(hit.rule.id)
                found.append(hit)
        return found[:self.cfg.get("documents.max_charges", 4)]

    def _pick_kind(self, dictation: str) -> str:
        """Вид документа по надиктовке.

        Спрашивать отдельно незачем: слова «иск», «жалоба», «суд» в
        надиктовке однозначно указывают на судопроизводство, всё
        остальное — прокурорское дело.
        """
        lowered = dictation.lower()
        court = ("иск", "исковое", "жалоб", "ходатайств", "апелляц",
                 "кассацион", "в суд", "судопроизводств")
        return "court" if any(w in lowered for w in court) else "proceedings"

    def _make_document(self) -> None:
        with self._busy:
            dictation = "\n".join(self._dictation).strip()
            if not dictation:
                self.overlay.sig_answer.emit("Нечего оформлять: надиктовки нет.")
                return

            kind = self._document_kind or self._pick_kind(dictation)
            templates = self.templates.load(kind)
            log.info("Документ «%s»: начинаю, надиктовка %d символов",
                     self._document_form or kind, len(dictation))
            form = self._document_form
            # Статьи из базы — в контекст документа. Раньше документ
            # писался только из образцов и надиктовки, и ни один номер
            # статьи не сверялся с законом: в бумагу уходило то, что
            # модель помнит, а не то, что написано в кодексе. Для
            # прокурорского дела и иска статья — основание, на неверной
            # дело разваливается.
            # Шаги 1-2: дело разбирается на эпизоды, под каждый ищется
            # своя норма. Общий поиск по надиктовке приносит «вообще
            # про задержание» и теряет половину нарушений.
            episodes = self._split_episodes(dictation)
            episode_text = episodes_for_prompt(episodes) if episodes else ""

            self._status("thinking", "Сверяю статьи с базой...")
            law_hits = self._document_law(dictation, form)
            # Нормы эпизодов — в общий контекст: на них документ и
            # опирается.
            for episode in episodes:
                for hit in episode.hits:
                    if hit.rule.id not in {h.rule.id for h in law_hits}:
                        law_hits.append(hit)
            law_context = ""
            if law_hits:
                try:
                    law_context = self.retriever.build_context(
                        law_hits,
                        max_article_chars=self.cfg.get(
                            "documents.max_article_chars", 1200),
                        max_context_chars=self.cfg.get(
                            "documents.max_law_chars", 5000))
                    self._log("База", "в документ: " + ", ".join(
                        h.rule.number for h in law_hits[:6]))
                except Exception:
                    log.exception("Выдержки для документа не собраны")

            # Статьи обвинения подбираем сами: модель из выдержек
            # выбирала не ту норму.
            charge_hits = self._charges(dictation)
            charges = "\n".join(
                f"  {h.rule.citation()} — {h.rule.gist(120)}"
                for h in charge_hits)
            if charge_hits:
                self._log("База", "статьи обвинения: " + ", ".join(
                    h.rule.number for h in charge_hits))
                law_hits = law_hits + [h for h in charge_hits
                                       if h not in law_hits]

            self._status("thinking",
                         f"Составляю: {(form or templates.title).lower()}...")
            self.overlay.sig_answer_chunk.emit(
                "Шаг 3 из 5: пишу черновик...\n\n")

            messages = [Message("system", document_prompt(
                templates.few_shot(
                    limit=self.cfg.get("documents.samples_in_prompt", 2),
                    max_chars=self.cfg.get("documents.max_sample_chars", 7000),
                    # Выбранная бумага важнее надиктовки: человек
                    # прямо сказал, что пишет определение о принятии
                    # иска, а не угадывать это по фактам дела.
                    like=f"{form} {dictation}" if form else dictation),
                templates.skeleton, dictation, form=form,
                # Правила подсудности нужны только судебным бумагам —
                # прокурорскому делу они лишние в промпте.
                court=(kind == "court"), law=law_context,
                charges=charges, episodes=episode_text))]

            self.overlay.begin_answer()
            pieces: list[str] = []
            started = time.monotonic()
            try:
                for chunk in self.llm.stream_chat(
                        messages, model=self._model_for("document")):
                    pieces.append(chunk)
                    self.overlay.sig_answer_chunk.emit(chunk)
            except LLMUnavailable as exc:
                log.error("Документ не составлен: %s", exc)
                self._status("error", str(exc))
                return

            document = "".join(pieces).strip()

            # Сверка статей до показа результата. Документ — не реплика
            # в задержании: здесь важнее правильность, чем секунды,
            # поэтому при неверной статье переписываем заново. Проверено:
            # 14B ставила УК 6.1 («телесные повреждения») за то, что
            # сотрудник не зачитал права, хотя нужная норма лежала
            # в выдержках рядом.
            if law_hits:
                wrong = check_answer(self.retriever, document, law_hits)
                if wrong:
                    log.warning("Документ переписывается: статьи %s", wrong)
                    self._status("thinking",
                                 "Статья не сошлась с базой — переписываю...")
                    # Первый вариант уже напечатан: без замены
                    # человек прочитает два документа подряд и не
                    # поймёт, какой из них настоящий.
                    self.overlay.sig_replace.emit(
                        document,
                        "[статья не сошлась с базой — переписываю]\n\n")
                    retry = messages + [
                        Message("assistant", document),
                        Message("user", DOCUMENT_RETRY.format(
                            numbers=", ".join(wrong))),
                    ]
                    try:
                        second: list[str] = []
                        for chunk in self.llm.stream_chat(
                                retry, model=self._model_for("document")):
                            second.append(chunk)
                            self.overlay.sig_answer_chunk.emit(chunk)
                        rewritten = "".join(second).strip()
                        if rewritten:
                            # Берём переписанный только если он не хуже:
                            # иногда вторая попытка ломает структуру.
                            still = check_answer(self.retriever, rewritten,
                                                 law_hits)
                            if len(still) < len(wrong):
                                document = rewritten
                    except LLMUnavailable as exc:
                        log.error("Перезапись документа не удалась: %s", exc)

            # Незаполненные поля — не брак, а сигнал: их видно сразу,
            # и пользователь допишет сам вместо того, чтобы искать
            # подставленный чужой номер паспорта.
            notes: list[str] = []
            gaps = document.count("[НЕ УКАЗАНО]")
            if gaps:
                # Просто «полей без данных: 6» человеку ничего не
                # говорит: найти их в готовой бумаге тяжело, а что
                # вписывать — непонятно. Называем каждое место.
                fields = unfilled_fields(document)
                listed = "".join(f"\n      • {name}"
                                 for name in fields)
                more = gaps - len(fields)
                if more > 0:
                    listed += f"\n      • и ещё {more}, ищите по метке"
                notes.append(
                    "Вы этого не говорили, поэтому бот не стал "
                    "придумывать. Впишите сами перед подачей "
                    f"({gaps}):{listed}")

            # Шаги 4-5: вычитка черновика и правка по замечаниям.
            # Проверено: за один проход модель либо пишет, либо
            # проверяет. Отдельными проходами она находит у себя
            # пропущенные эпизоды и чужие статьи.
            notes_from_review = self._review_document(
                document, episodes, law_context, law_hits)
            if notes_from_review:
                self._status("thinking", "Вношу правки...")
                self.overlay.sig_replace.emit(
                    document, "[вычитка нашла замечания — правлю]\n\n")
                remarks = "\n".join(f"- {n}" for n in notes_from_review)
                try:
                    revised = self.llm.chat(
                        messages + [
                            Message("assistant", document),
                            Message("user",
                                    CASE_REVISE.format(notes=remarks)),
                        ],
                        num_predict=self.cfg.get("documents.predict", 1400),
                        model=self._model_for("document"))
                    # Правка обязана улучшать. Проверено: по
                    # замечаниям вычитки модель сменила вид документа
                    # с «о принятии» на «об отказе в принятии» и
                    # приписала статьи кодекса, которого нет.
                    better, why = revision_is_better(
                        document, revised,
                        check_answer(self.retriever, document, law_hits),
                        check_answer(self.retriever, revised, law_hits))
                    if better:
                        document = revised.strip()
                        self.overlay.sig_answer_chunk.emit(document)
                    else:
                        log.warning("Правка отклонена: %s", why)
                        self._log("Проверка", f"правка отклонена: {why}")
                        self.overlay.sig_answer_chunk.emit(document)
                except LLMUnavailable as exc:
                    log.warning("Правка по замечаниям не удалась: %s", exc)

            # Промпт запрещает выдумывать даты, время и номера, но
            # 4B-модель запрет нарушает: сама проставляет правдоподобное
            # «арестован в 14:15». По времени считают процессуальные
            # сроки, и перепроверять его никто не станет.
            # Не просто предупреждаем, а вычёркиваем: бумага уходит
            # в дело целиком, и выдуманное время останется в ней, даже
            # если приписка внизу замечена. На месте вычеркнутого
            # встаёт [НЕ УКАЗАНО] — та же метка, которой промпт велит
            # помечать незаполненное.
            shown = document
            document, invented = redact_invented_facts(document, dictation)
            if document != shown:
                # Документ печатался по мере генерации, и вычеркнутое
                # осталось бы на экране: человек видел бы имя судьи из
                # образца там, где в тексте уже стоит [НЕ УКАЗАНО].
                self.overlay.sig_replace.emit(shown, document)

            # Статьи в документе сверяем с базой. Это последняя
            # преграда: промпт запрещает выдумывать номера, но запрет
            # — просьба, а не гарантия. В иске или деле неверная
            # статья означает, что бумагу вернут или дело развалится.
            if law_hits:
                # Номера статей и величины разводим по серьёзности:
                # неверная статья рушит дело, непроверенный срок —
                # повод сверить глазами. В одном списке они путали.
                bad_numbers = self.retriever.unsupported_citations(
                    document, law_hits)
                bad_figures = [f for f in self.retriever.unsupported_figures(
                    document, law_hits) if f not in bad_numbers]
                if bad_numbers:
                    log.warning("В документе статьи вне базы: %s", bad_numbers)
                    notes.append(
                        "СТАТЬИ НЕТ В БАЗЕ: " + ", ".join(bad_numbers[:8])
                        + ". Это серьёзно: в законе штата такой нормы "
                        "нет или она о другом — проверьте номер.")
                    self._log("Проверка",
                              "статьи вне базы: " + ", ".join(bad_numbers[:6]))
                if bad_figures:
                    notes.append(
                        "СВЕРЬТЕ ЦИФРЫ: " + ", ".join(bad_figures[:8])
                        + ". Их нет в найденных нормах — возможно, взяты "
                        "из образца.")
                if not bad_numbers:
                    cited = self.retriever.cited_rules(document, law_hits)
                    if cited:
                        self._log("Проверка", "статьи сверены с базой: "
                                  + ", ".join(r.number for r in cited[:6]))

            # Документ уносят целиком — его и запоминаем под клавишу
            # копирования. Приписки про пустые поля идут ниже и в
            # буфер не попадут: их в бумаге быть не должно.
            self._last_answer = document
            if invented:
                log.warning("В документе данные вне надиктовки: %s", invented)
                # Повторы убираем: одно и то же имя модель вставляет
                # в шапку и в подпись, и список выглядел так, будто
                # бот написал его дважды.
                unique: list[str] = []
                for item in invented:
                    if item not in unique:
                        unique.append(item)
                notes.append(
                    "УБРАЛ ИЗ ДОКУМЕНТА — вы этого не говорили, а модель "
                    "взяла из образца: " + "; ".join(unique[:8])
                    + ". В тексте на их месте стоит [НЕ УКАЗАНО] — "
                    "впишите свои значения.")

            if notes:
                note = "\n\n[!] " + "\n[!] ".join(notes)
                self.overlay.sig_answer_chunk.emit(note)
                document += note

            self.history.record(
                session_id=self.sessions.current.id, mode="document",
                source=Source.SELF.title,
                question=f"Оформление: {templates.title}\n{dictation}",
                answer=document,
                latency_ms=int((time.monotonic() - started) * 1000))
            self._log("Система", f"{templates.title}: составлен, "
                                 f"пустых полей {gaps}")
            self._status("ready", "Документ готов")
            # Возвращаемся к быстрой модели заранее, фоном. Иначе за
            # переключение платит следующий вопрос: семь с половиной
            # секунд ожидания там, где обычно уходит полторы.
            self._submit(self._restore_fast_model)
            self._dictation = []

    # ------------------------------------------------- речь собеседника
    def _start_other(self) -> None:
        """Захват звука из GTA5.exe и Discord.exe, пока зажата клавиша.

        Пишутся только эти процессы: Process Loopback активируется по
        конкретному PID, поэтому системный звук физически не попадает
        в запись — это не проверка на нашей стороне, а свойство самого
        механизма (требование ТЗ, раздел 2).
        """
        if self._busy.locked():
            self._status("thinking", "Дождитесь ответа на предыдущий вопрос")
            return
        try:
            captured = self.other.start()
        except LoopbackError as exc:
            log.warning("Захват собеседника не начат: %s", exc)
            self._status("error", str(exc)[:90])
            self.overlay.sig_answer.emit(f"Не удалось начать запись:\n\n{exc}")
            return
        except Exception as exc:
            log.exception("Сбой захвата собеседника")
            self._status("error", f"Сбой захвата: {exc}")
            return

        self._ptt_started_at = time.monotonic()
        self._status("recording", "Пишу: " + ", ".join(captured))

    def _finish_other(self) -> None:
        if not self.other.is_recording:
            return
        held = time.monotonic() - self._ptt_started_at
        if held < self._min_ptt_s:
            self.other.cancel()
            self._status("idle", f"Слишком короткое нажатие ({held:.1f} с)")
            return

        try:
            audio = self.other.stop()
        except Exception as exc:
            log.exception("Сбой остановки захвата")
            self._status("error", f"Сбой захвата: {exc}")
            return

        if audio.size == 0:
            self._status("error", "Процессы не отдали звук за это время")
            return

        # Whisper работает на 16 кГц, а захват идёт на 48 — пересчитываем.
        source_rate = self.cfg.get("audio.loopback_sample_rate", 48000)
        target_rate = self.cfg.get("audio.sample_rate", 16000)
        if source_rate != target_rate:
            length = int(round(len(audio) * target_rate / source_rate))
            positions = np.linspace(0, len(audio) - 1, length)
            audio = np.interp(positions, np.arange(len(audio)),
                              audio).astype(np.float32)

        peak = float(np.max(np.abs(audio)))
        if peak < 0.002:
            self._status("idle", "Собеседник молчал — записана тишина")
            return

        self._status("transcribing", "Распознаю речь собеседника...")
        self._submit(self._run_pipeline, audio, Source.OTHER)

    # -------------------------------------------------------------- конвейер
    def _run_pipeline(self, audio, source: Source) -> None:
        with self._busy:
            try:
                transcript = self.stt.transcribe(
                    audio, self.cfg.get("audio.sample_rate", 16000))
            except Exception as exc:
                log.exception("Сбой распознавания")
                self._status("error", f"Распознавание не удалось: {exc}")
                return

            if transcript.is_empty:
                self._status("idle", "Речь не распознана — тишина или слишком тихо")
                return

            self._log(source.title, transcript.text)
            # Аудио держим до следующего вопроса: по хоткею переспроса
            # его можно пересчитать шире, не наговаривая заново.
            self._last_audio = (audio, source)

            if self.sessions.current.mode is Mode.DOCUMENT:
                # В режиме документа надиктовка копится, а не идёт
                # в модель: документ собирается из всего сказанного
                # разом, иначе каждая фраза породила бы свою бумагу.
                self._dictation.append(transcript.text)
                self._show_dictation()
                return

            self._answer(transcript.text, source)

    def _answer(self, question: str, source: Source) -> None:
        if not self._llm_ready:
            ready, reason = self.llm.check_ready()
            self._llm_ready = ready
            if not ready:
                self._status("error", reason)
                self.overlay.sig_answer.emit(
                    f"Вопрос распознан:\n«{question}»\n\n"
                    f"Но ответить не могу: {reason}"
                )
                return

        session = self.sessions.current

        detailed = wants_detail(question)
        hits = []
        context = ""
        if not self._rag_ready:
            # Поиск мог отвалиться на старте — например, Ollama тогда
            # ещё не поднялась. Раньше флаг ставился один раз при
            # прогреве и больше не пересматривался: бот до самого
            # закрытия отвечал «в базе не нашлось», хотя база на месте
            # и всё уже работает. Пробуем ещё раз при каждом вопросе.
            self._check_index()

        if self._rag_ready:
            if detailed and self._last_hits:
                # «Расскажи подробнее» само по себе не содержит ничего,
                # что можно искать. Берём статьи прошлого ответа —
                # именно их и просят раскрыть.
                hits = self._last_hits
                self._status("thinking", "Разворачиваю прошлый ответ...")
            else:
                self._status("thinking", "Ищу статьи в базе...")
                try:
                    hits = self.retriever.retrieve(
                        question, limit=self._mode_setting(
                            session.mode, "top_k",
                            self.cfg.get("retrieval.top_k", 6)))
                except Exception:
                    log.exception("Сбой поиска по базе")
                if getattr(self.retriever, "last_degraded", ""):
                    # Поиск отработал, но без смыслового слоя: ответ
                    # будет заметно хуже обычного, и человек должен
                    # знать причину, а не гадать.
                    self._log("Система", "Смысловой поиск недоступен — "
                                         "ищу только по словам")
            try:
                # Выдержки обязаны влезть в окно модели: у облегчённого
                # профиля оно меньше, и промпт экзамена однажды вышел за
                # край — Ollama ответила 400, а человек увидел «ошибка
                # обращения к модели» посреди экзамена.
                budget = self.retriever.context_budget(
                    self.cfg.profile["llm"].get("num_ctx", 0),
                    self._mode_setting(session.mode, "num_predict",
                                       self.cfg.get("llm.num_predict", 320)),
                    len(system_prompt(session.mode, legal_context="",
                                      rag_enabled=self._rag_ready)))
                # На экзамене статью нельзя обрезать: спрашивают как раз
                # полноту перечня, а обрубленное перечисление стоит балла.
                context = self.retriever.build_context(
                    hits, max_article_chars=self._mode_setting(
                        session.mode, "max_article_chars", 0),
                    max_context_chars=budget)
            except Exception:
                log.exception("Сбой сборки контекста")
            if hits:
                self._log("База", ", ".join(
                    f"{h.rule.number} ({h.why()})" for h in hits[:4]))
            else:
                # Пустая выдача — редкость и почти всегда признак сбоя,
                # а не отсутствия нормы. Записываем сам запрос: без него
                # разобраться потом не по чему.
                self._log("База", f"поиск не дал ни одной статьи: "
                                  f"«{question[:70]}»")

        checklist_text = ""
        if session.mode is Mode.LAWYER and self._checklist is not None:
            checklist_text = self._checklist.as_prompt()
        prompt = system_prompt(session.mode, legal_context=context,
                               rag_enabled=self._rag_ready,
                               detailed=detailed, checklist=checklist_text,
                               shape=_answer_shape(question),
                               lawyer_role=self._lawyer_role)
        messages = session.build_messages(prompt)
        messages.append(Message("user", question))

        self._status("thinking", "Формулирую ответ...")
        self.overlay.begin_answer()
        # Распознанный вопрос — строкой над ответом, а не только в логе
        # внизу. Лог пользователь скрывает, когда тот разрастается, и
        # именно так мимо него прошло «хладность» вместо «халатности»:
        # ответ был про отсутствие нормы, а причина — в одной букве.
        self.overlay.sig_answer_chunk.emit(f"❯ {question}\n\n")
        started_at = time.monotonic()

        # Потолок длины ответа. На CPU время генерации прямо
        # пропорционально числу токенов, поэтому режим, которому длинный
        # ответ не нужен, получает ответ быстрее — без ущерба для поиска.
        budget = 0 if detailed else self._mode_setting(
            session.mode, "num_predict", 0)

        # Генерация с детерминированной проверкой ссылок. Обёртка одна
        # на бота и на проверочные утилиты: раньше логика жила только
        # здесь, и tools/ мерили не то, что получает пользователь.
        # Статьи чек-листа модель тоже видит, значит ссылка на них
        # законна — иначе проверка ругалась бы на собственный промпт.
        allowed = list(hits) + self._checklist_hits()
        try:
            checked = generate_checked(
                self.llm, messages, self.retriever, allowed,
                num_predict=budget,
                # Вопросы считает быстрая модель. Если до этого
                # оформлялся документ на 14B, здесь произойдёт обратное
                # переключение — тоже само.
                model=self._model_for("answer"),
                retry=self._mode_setting(session.mode,
                                         "retry_on_invented", False),
                redact=self._mode_setting(session.mode,
                                          "redact_invented", False),
                on_chunk=self.overlay.sig_answer_chunk.emit,
                on_retry=self._on_invented_retry)
        except LLMUnavailable as exc:
            log.error("LLM недоступна: %s", exc)
            self._llm_ready = False
            self._status("error", str(exc))
            self.overlay.sig_answer.emit(f"Ошибка обращения к модели:\n{exc}")
            return

        # Если модель всё же воспроизвела служебный блок — вырезаем.
        answer = strip_service_blocks(checked.text)
        invented = checked.invented

        # Перечень со ссылкой в каждой строке читается как таблица.
        # Просьбы в промпте не хватило: пункты действительно взяты из
        # разных пунктов закона, и модель честно ссылается на каждый.
        # Сворачиваем сами — смысл не меняется, меняется только объём.
        # Повторы одной и той же ссылки внутри ответа — читать тяжело,
        # произнести вслух невозможно.
        answer = dedupe_citations(answer)
        folded = fold_citations(answer)
        if folded != answer:
            self.overlay.sig_replace.emit(answer, folded)
            answer = folded

        if invented:
            log.warning("Модель назвала статьи вне контекста: %s", invented)
            # Проверка ловит и номера статей, и суммы со сроками,
            # поэтому формулировка общая: «в выдержках нет вот этого».
            warning = ("\n\n[!] ПРОВЕРЬТЕ ВРУЧНУЮ: в найденных статьях нет "
                       + ", ".join(invented) +
                       " — ассистент мог это выдумать.")
            answer += warning
            self.overlay.sig_answer_chunk.emit(warning)
            self._log("Проверка", f"вне контекста: {', '.join(invented)}")

        # То, что сказала сама модель, — отдельно от того, что видит
        # человек. В контекст диалога уходит только первое: служебный
        # блок «о чём статья» модель принимала за образец своего же
        # ответа и на следующем вопросе дописывала его сама. В ленте
        # тогда появлялось два блока подряд.
        spoken = answer
        # По клавише копирования уносится именно это: пересказ статьи
        # под ответом человеку в игре не нужен.
        self._last_answer = spoken
        answer += self._explain_cited(answer, allowed)

        articles = self.retriever.articles_mentioned(hits)
        # Считаем документы, на которые сослался готовый ответ, а не те,
        # что просто нашлись: нашлось шесть, пригодилась одна статья —
        # её документ и есть ходовой. Дальше он получает небольшую фору
        # в выдаче при прочих равных.
        cited = {r.document for r in self.retriever.cited_rules(answer, allowed)}
        if cited:
            self.legal_db.note_usage(cited)
            self.retriever.forget_usage()

        session.add(source, question, spoken, articles=articles)
        self.history.record(
            session_id=session.id, mode=session.mode.value,
            source=source.title, question=question, answer=answer,
            articles=articles, invented=invented,
            latency_ms=int((time.monotonic() - started_at) * 1000))
        if not detailed:
            self._last_question, self._last_hits = question, hits
        if self._checklist is not None:
            covered = self._checklist.mark_covered(answer)
            for title in covered:
                self._log("Чек-лист", f"пройдено: {title}")
            following = self._checklist.next_question()
            if following is not None:
                # Подсказка идёт следом за ответом: адвокат не
                # должен вспоминать, о чём спрашивать дальше.
                self.overlay.sig_answer_chunk.emit(
                    "\n\n── ДАЛЬШЕ СПРОСИТЬ ──\n"
                    f"«{following.spoken_question()}»")
        self._log("Ассистент", answer)
        self._status("ready", self._hotkey_hint())

        if session.needs_summary:
            self._compress(session)

    def _show_cases(self) -> None:
        """Показывает список дел: клик мышкой открывает любое.

        Дело — это сессия: она начинается сбросом контекста и живёт,
        пока разбирается одна ситуация. Раньше вернуться к прошлому
        разбору было нельзя вовсе, а прокурорское дело за один присест
        не закрывается.
        """
        # Повторное нажатие закрывает список и возвращает к своим
        # вопросам — отдельный хоткей «назад» заводить незачем.
        if self._showing_cases:
            self._showing_cases = False
            self.overlay.set_cases_tab(False)
            self.overlay.show_transcript()
            self._status("idle", self._hotkey_hint())
            return

        cases = self.history.cases(limit=self.cfg.get("history.cases", 25))
        if not cases:
            self.overlay.sig_answer.emit(
                "Дел пока нет — они появятся после первых вопросов.")
            return

        rows = ["<div style='font-size:15px; font-weight:600; "
                "margin-bottom:8px'>ДЕЛА</div>"]
        for case in cases:
            title = _escape(case.title)
            rows.append(
                f"<div style='margin-bottom:9px'>"
                f"<a href='case:{case.session_id}' "
                f"style='color:#93c5fd; text-decoration:none'>{title}</a>"
                f"<br><span style='color:#9ca3af; font-size:11px'>"
                f"{case.when} · {case.count} зап. · {_mode_title(case.mode)}"
                f"</span></div>")
        rows.append("<div style='color:#9ca3af; font-size:11px; "
                    "margin-top:10px'>Щёлкните по делу мышкой. "
                    "Вернуться к текущему — «← Назад» вверху или "
                    f"{self.cfg.hotkeys.get('cases')}.</div>")

        self._showing_cases = True
        self.overlay.set_cases_tab(True)
        self.overlay.show_html("".join(rows))
        self._status("idle", f"Дел в истории: {len(cases)}")

    # ------------------------------------------------------- поиск по делам
    def _toggle_search(self) -> None:
        """Открывает и закрывает строку поиска по делам.

        Дел за неделю работы с прокуратурой набирается столько, что
        листать список бессмысленно: нужен номер постановления, а он
        лежит внутри самого документа.
        """
        if self._search_active:
            self._close_search()
            return
        self._search_active = True
        self._search_query = ""
        # Пока строка открыта, буквы забирает хук: окно ассистента
        # фокуса не получает, и обычное поле ввода в нём не работает —
        # набранное уходило бы в игровой чат.
        self.hotkeys.capture_text(True)
        self._render_search()

    def _close_search(self) -> None:
        if not self._search_active:
            return
        self._search_active = False
        self._search_query = ""
        self._search_hits = []
        self.hotkeys.capture_text(False)
        self._showing_cases = False
        self.overlay.set_cases_tab(False)
        self.overlay.show_transcript()
        self._status("idle", self._hotkey_hint())

    def _search_type(self, char: str, *, back: bool = False) -> None:
        if not self._search_active:
            return
        if back:
            self._search_query = self._search_query[:-1]
        else:
            self._search_query += char
        self._render_search()

    def _open_first_found(self) -> None:
        """Enter — открыть первое найденное дело."""
        if not self._search_active:
            return
        if not self._search_hits:
            self._status("idle", "По запросу ничего не нашлось")
            return
        session_id = self._search_hits[0].session_id
        self._close_search()
        self._open_case(session_id)

    def _render_search(self) -> None:
        """Рисует строку поиска и то, что под неё подошло."""
        query = self._search_query
        try:
            cases = self.history.find_cases(
                query, limit=self.cfg.get("history.cases", 25))
        except Exception:
            log.exception("Поиск по делам не отработал")
            cases = []
        self._search_hits = cases

        typed = _escape(query) or (
            "<span style='color:#6b7280'>номер или слово из дела</span>")
        rows = ["<div style='font-size:15px; font-weight:600; "
                "margin-bottom:6px'>ПОИСК ПО ДЕЛАМ</div>",
                f"<div style='color:#f3f4f6; font-size:15px; "
                f"margin-bottom:10px'>&gt; {typed}<span "
                f"style='color:#93c5fd'>|</span></div>"]

        if not cases:
            rows.append("<div style='color:#9ca3af'>Ничего не нашлось.</div>")
        for position, case in enumerate(cases):
            # Первое дело подсвечено: именно оно откроется по Enter.
            colour = "#bfdbfe" if position == 0 else "#93c5fd"
            rows.append(
                f"<div style='margin-bottom:9px'>"
                f"<a href='case:{case.session_id}' "
                f"style='color:{colour}; text-decoration:none'>"
                f"{_escape(case.title)}</a>"
                f"<br><span style='color:#9ca3af; font-size:11px'>"
                f"{case.when} &middot; {case.count} зап. &middot; "
                f"{_mode_title(case.mode)}</span></div>")

        rows.append("<div style='color:#9ca3af; font-size:11px; "
                    "margin-top:10px'>Enter — открыть первое, "
                    "Esc — закрыть. Можно щёлкнуть мышкой.</div>")

        self._showing_cases = True
        self.overlay.set_cases_tab(True)
        self.overlay.show_html("".join(rows))
        self._status("idle", f"Найдено дел: {len(cases)}")

    def _open_case(self, session_id: str) -> None:
        """Разворачивает одно дело целиком."""
        entries = self.history.session(session_id)
        if not entries:
            self.overlay.sig_answer.emit("Дело пустое.")
            return

        parts = []
        for entry in entries:
            parts.append(f"[{entry.timestamp[11:16]}] {entry.source}: "
                         f"{entry.question}\n{entry.answer}")
        text = ("\n\n" + "─" * 34 + "\n\n").join(parts)

        self._showing_cases = False
        self.overlay.set_cases_tab(False)
        self.overlay.show_transcript()
        self.overlay.sig_answer.emit(
            f"ДЕЛО ОТ {entries[0].timestamp[:10]}\n"
            f"записей: {len(entries)}\n\n{text}")
        self._status("idle", "Открыто дело · назад к списку — «☰ Дела»")

    def _mark_wrong(self) -> None:
        """Помечает последний ответ как неверный.

        Раньше единственным способом зафиксировать промах было
        написать разработчику, и запоминались только те случаи, что
        пользователь успел пересказать. Отмеченные промахи копятся в
        истории и разбираются потом: python tools/history.py --wrong.
        """
        entry = self.history.mark_wrong()
        if entry is None:
            self._status("idle", "Пока нечего помечать — ответов не было")
            return
        self._log("Проверка", f"помечено неверным: {entry.question[:60]}")
        self._status("ready",
                     "Ответ помечен неверным — разберём его позже. "
                     + self._hotkey_hint())

    def _on_invented_retry(self, numbers: list[str],
                           discarded: str = "") -> None:
        """Убирает забракованный ответ и говорит, почему он переписан.

        Ответ печатается по мере генерации, поэтому к моменту
        проверки он уже на экране. Без этой замены второй ответ
        дописывался следом, и человек читал два ответа подряд —
        именно это и всплыло на настоящем экзамене.
        """
        self._log("Проверка", "статья вне базы — переспрашиваю модель")
        note = "[перепроверяю: названа статья, которой нет в базе]\n\n"
        if discarded.strip():
            self.overlay.sig_replace.emit(discarded.strip(), note)
        else:
            self.overlay.sig_answer_chunk.emit("\n\n" + note)

    def _model_for(self, task: str) -> str:
        """Какая модель считает эту задачу, с освобождением памяти под неё.

        Тяжёлые задачи — прокурорское дело, судебный документ,
        заключение для суда — получают 14B: она заметно лучше держит
        структуру и не теряет ссылки. Быстрые вопросы в задержании
        остаются на 7B.

        Одновременно они в шестнадцать гигабайт не помещаются: 14B это
        десять с лишним плюс Whisper и буферы CUDA. Поэтому перед
        переключением ненужная выгружается. Плата — секунд пятнадцать
        на загрузку, и приходится она ровно на те задачи, где человек
        и так диктует минуту.
        """
        model = self.cfg.get(f"models.{task}")
        if not model:
            return self._active_model

        # Тяжёлая модель нужна только если она влезет целиком. Иначе
        # Ollama держит часть слоёв в системной памяти, и генерация
        # идёт по токену в секунду: документ в пять проходов тогда
        # считается минутами и выглядит зависшим. Лучше написать его
        # на быстрой модели, чем не написать вовсе.
        needs = self.cfg.get(f"models.{task}_needs_mb", 0)
        if needs and not _same_model(model, self._active_model):
            free = free_vram_mb()
            # Память, которую сейчас держат наши же модели, вернётся
            # при выгрузке — её тоже считаем свободной.
            for name, size in self.llm.loaded_sizes().items():
                if not name.startswith("bge-m3"):
                    free = (free or 0) + size // (1024 * 1024)
            if self.stt.is_loaded:
                free = (free or 0) + self.cfg.get("models.whisper_mb", 1500)
            if free is not None and free < needs:
                fallback = self.cfg.get("models.answer") or self._active_model
                log.warning("Для %s нужно %d МБ, свободно %d — беру %s",
                            model, needs, free, fallback)
                self._log("Система",
                          f"Памяти мало ({free} МБ) — считаю на {fallback}")
                model = fallback

        if model == self._active_model:
            return model

        for loaded in self.llm.loaded_models():
            # Сравниваем полное имя, а не часть до двоеточия: у 7B и 14B
            # она одинаковая («qwen2.5»), и по ней 7B никогда не
            # выгружалась — модели пытались ужиться в одной карте.
            # bge-m3 не трогаем: без него встанет поиск по смыслу,
            # а весит он меньше гигабайта.
            if loaded.startswith("bge-m3"):
                continue
            if _same_model(loaded, model):
                continue
            self.llm.unload(loaded)

        # Освобождаем и Whisper: 14B занимает 9.5 ГБ, распознавание ещё
        # несколько, и вместе с буферами CUDA это выходит за шестнадцать.
        # К моменту генерации документа диктовка уже закончена, а к
        # следующему нажатию PTT модель поднимется заново за три секунды.
        if not _same_model(model, self.llm.options.model) and self.stt.is_loaded:
            self.stt.unload()
            self._log("Система", "Распознавание выгружено — освобождаю память")

        self._log("Система", f"Переключаюсь на модель {model}")
        self._status("thinking", f"Загружаю {model} — это до полуминуты...")
        self._active_model = model
        return model

    def _restore_fast_model(self) -> None:
        """Возвращает в память быструю модель после тяжёлой задачи.

        Статус после себя обязана вернуть на место: переключение
        моделей ставит «Загружаю...», и человек видел его уже поверх
        готового документа — выглядело так, будто бот всё ещё думает.
        """
        answer_model = self.cfg.get("models.answer")
        if not answer_model or _same_model(answer_model, self._active_model):
            return
        was_ready = self._last_answer
        try:
            self._model_for("answer")
            self.llm.chat([Message("user", "готов")], num_predict=1,
                          model=answer_model)
            log.info("Быстрая модель возвращена в память")
        except LLMUnavailable as exc:
            log.warning("Не удалось вернуть быструю модель: %s", exc)
        finally:
            self._status("ready" if was_ready else "idle",
                         "Документ готов" if was_ready
                         else self._hotkey_hint())

    def _mode_setting(self, mode, key: str, default):
        """Настройка режима из секции modes конфига, иначе умолчание.

        Режимы требуют разного: в задержании ответ должен быть коротким
        и мгновенным, на экзамене — связным и полным. Один набор
        параметров на всех означает, что один из режимов всегда работает
        хуже, чем мог бы.

        Ноль у числовой настройки означает «как в общей секции», а не
        «ноль». Это стоило двух потерянных ответов: в конфиге у обычного
        режима стоял top_k: 0 с комментарием «0 = как в retrieval»,
        код брал ноль буквально и искал ноль статей, а пользователь
        видел честное «в базе не нашлось» вместо ответа.

        Булево значение под это правило не попадает: False — полноценный
        ответ «выключено», а не пропуск настройки.
        """
        value = self.cfg.get(f"modes.{mode.value}.{key}")
        if value is None:
            return default
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return default if value == 0 else value
        return value

    def _retry_question(self) -> None:
        """Пересчитывает последнюю запись более точным декодером.

        Переспрос нужен не потому, что модель плохо ответила, а потому
        что Whisper плохо расслышал: «хладность» вместо «халатности»,
        «обозвалка» вместо «обозвал». Наговаривать заново не нужно —
        аудио сохранено, меняется только ширина луча декодера.
        """
        if self._last_audio is None:
            self._status("idle", "Нечего переспрашивать — вопроса ещё не было")
            return
        audio, source = self._last_audio

        with self._busy:
            self._status("thinking", "Переспрашиваю — слушаю запись внимательнее...")
            try:
                transcript = self.stt.transcribe(
                    audio, self.cfg.get("audio.sample_rate", 16000),
                    beam_size=self.cfg.get("stt.retry_beam_size", 5))
            except Exception as exc:
                log.exception("Переспрос не удался")
                self._status("error", f"Переспрос не удался: {exc}")
                return

            if transcript.is_empty:
                self._status("idle", "При переспросе речь не распознана")
                return

            self._log("Переспрос", transcript.text)

        # Сессию не трогаем: прошлый разбор вопроса остаётся в истории,
        # иначе непонятно, почему ответ вдруг сменился.
        self._answer(transcript.text, source)

    def _explain_cited(self, answer: str, hits: list) -> str:
        """Дописывает под ответом короткую суть названных статей.

        Сначала здесь печаталась статья целиком. Для редкой нормы это
        было правильно, но на деле ответ раздувался до экрана текста,
        который в разговоре с сотрудником никто не читает: нужна суть,
        а дословная формулировка — только когда на неё ссылаются вслух.
        Поэтому полный текст ушёл под отдельную клавишу, а здесь
        осталась строка на статью.

        Возвращает дописанное — вызывающий добавляет это к ответу,
        чтобы пояснение попало и в историю.
        """
        rules = self.retriever.cited_rules(answer, hits)
        fallback = False
        if not rules:
            # Модель бывает забывает поставить ссылку — 7B теряет её
            # примерно в каждом третьем ответе. Ответ без номера
            # бесполезен: по нему нельзя ни возразить сотруднику, ни
            # проверить сам ответ. Показываем лучшую найденную статью
            # и честно помечаем, что сослался поиск, а не модель.
            if not hits or _is_refusal(answer):
                # Бот честно сказал, что ответа в базе нет. Подсовывать
                # к такому ответу «скорее всего эта статья» — худшее из
                # возможного: человек видит отказ и рядом чужую норму,
                # которая к вопросу отношения не имеет.
                self._last_cited = []
                return ""
            rules, fallback = [hits[0].rule], True

        limit = self.cfg.get("retrieval.explain_articles", 4)
        # Статьи запоминаем всегда: по клавише их разворачивают
        # целиком, в том числе и на экзамене.
        self._last_cited = list(rules[:limit])

        # На экзамене блок лишний: ответ там и есть объяснение своими
        # словами, а пересказ рядом с ним только отнимает время.
        if self.sessions.current.mode is Mode.EXAM:
            return ""

        gist_chars = self.cfg.get("retrieval.gist_chars", 160)
        full = self.cfg.hotkeys.get("full_article", "ctrl+alt+s")

        head = ("── СКОРЕЕ ВСЕГО ЭТА СТАТЬЯ ──" if fallback
                else "── О ЧЁМ СТАТЬЯ ──")
        lines = ["", "", head]
        for rule in self._last_cited:
            line = f"{rule.citation()} — {rule.gist(gist_chars)}"
            if rule.sanction:
                line += f"\n   Санкция: {rule.sanction}"
            lines.append(line)
        lines.append("")
        lines.append(f"Текст статьи целиком — {full}")
        block = "\n".join(lines).rstrip()
        self.overlay.sig_answer_chunk.emit(block)
        return block

    def _copy_answer(self) -> None:
        """Кладёт последний ответ в буфер обмена.

        Окно намеренно не получает фокус, поэтому выделить мышью и
        нажать ctrl+c в нём нельзя — нажатие уходит в игру. Готовый
        документ без этого пришлось бы перепечатывать руками.
        """
        text = (self._last_answer or "").strip()
        if not text:
            self._status("idle", "Копировать нечего — ответа ещё не было")
            return
        try:
            clipboard.set_text(text)
        except Exception as exc:
            log.exception("Ответ не скопирован")
            self._status("error", f"Буфер занят: {exc}")
            return
        self._log("Система", f"Скопировано в буфер: {len(text)} символов")
        self._status("ready", f"Скопировано ({len(text)} символов) — "
                              "вставляйте через ctrl+v")

    def _paste_text(self) -> None:
        """Берёт текст из буфера вместо голоса.

        Распознавание иногда обрывает длинную фразу на середине, а
        часть материала и вовсе приходит текстом — из форума, из чата,
        от судьи. Набирать это голосом заново бессмысленно.

        Текст попадает туда же, куда попала бы речь: в режиме
        оформления — в надиктовку, в остальных — вопросом.
        """
        try:
            text = clipboard.get_text()
        except Exception as exc:
            log.exception("Буфер обмена не прочитан")
            self._status("error", f"Буфер не прочитан: {exc}")
            return

        text = " ".join((text or "").split())
        if not text:
            self._status("idle", "В буфере пусто — скопируйте текст (ctrl+c)")
            return

        limit = self.cfg.get("clipboard.max_paste_chars", 4000)
        if len(text) > limit:
            # Длинный текст — это, скорее всего, скопированная страница
            # форума, а для неё есть импорт в базу. Молча обрезать
            # нельзя: половина вопроса хуже отсутствия вопроса.
            self._status("idle",
                         f"В буфере {len(text)} символов — это похоже на "
                         f"страницу форума. Для базы: "
                         f"{self.cfg.hotkeys.get('import_laws')}")
            return

        self._log("Вставка", text[:90])
        if self.sessions.current.mode is Mode.DOCUMENT:
            self._dictation.append(text)
            self._show_dictation()
            return
        self._answer(text, Source.SELF)

    def _show_full_article(self) -> None:
        """Хоткей: развернуть названные статьи дословно.

        Отдельно от «подробнее»: тот переспрашивает модель, а здесь
        ничего не генерируется — печатается текст из базы. В разговоре
        с сотрудником нужна именно дословная формулировка, и ждать
        ответа модели ради неё незачем.
        """
        if not self._last_cited:
            self._status("idle", "Сначала задайте вопрос")
            return

        cap = self.cfg.get("retrieval.explain_chars", 1600)
        lines = ["", "", "── СТАТЬЯ ЦЕЛИКОМ ──"]
        for rule in self._last_cited:
            body = rule.text
            if rule.sanction:
                body += f"\nСанкция: {rule.sanction}"
            if len(body) > cap:
                body = body[:cap].rsplit(" ", 1)[0] + "..."
            lines.append(rule.citation())
            lines.append(body)
            lines.append("")
        self.overlay.sig_answer_chunk.emit("\n".join(lines).rstrip())
        self._status("ready", self._hotkey_hint())

    def _compress(self, session) -> None:
        """Мягкая суммаризация старой части сессии (ТЗ 5.7)."""
        def summarize(transcript: str) -> str:
            return self.llm.chat([
                Message("system", SUMMARY_PROMPT),
                Message("user", transcript),
            ])

        self._status("thinking", "Сворачиваю ранние реплики в конспект...")
        if session.compress(summarize):
            self._log("Система", "Ранняя часть сессии свёрнута в конспект")
        self._status("ready", self._hotkey_hint())
