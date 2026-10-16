"""
Клиент Ollama (ТЗ 6).

Стримим ответ токенами: на CPU-профиле генерация идёт десятки секунд,
и ждать её целиком, показывая пустой оверлей, неприемлемо — текст должен
появляться по мере готовности.

Выбор карты: сам Ollama — отдельная служба, и переменные окружения нашего
процесса на неё не влияют. Поэтому слой, который должен считаться на GPU,
задаётся параметром num_gpu в запросе, а карта — переменной окружения
самой службы Ollama (см. README, раздел про пиннинг).
"""
from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator

import requests

log = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    """Сервер Ollama не отвечает или не отдал модель."""


@dataclass
class Message:
    role: str            # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMOptions:
    model: str
    num_ctx: int = 8192
    num_gpu: int = 0
    temperature: float = 0.2
    top_p: float = 0.9
    timeout: int = 180
    # Дедлайн на весь потоковый ответ, включая загрузку модели.
    # Отдельный от timeout: тот отмеряет паузу между пакетами, а
    # висящий запрос пакеты как раз шлёт.
    stream_timeout: int = 300
    stop: list[str] = field(default_factory=list)
    # Потолок длины ответа в токенах. На CPU генерация упирается именно
    # в их число: ответ на 200 токенов считается втрое дольше, чем на 70,
    # при одинаковом поиске и одинаковом промпте. Поэтому режимы,
    # которым длинный ответ не нужен, задают свой потолок.
    num_predict: int = 0          # 0 = без ограничения
    # Сколько держать модель в памяти после ответа. По умолчанию Ollama
    # выгружает её через пять минут, и следующий вопрос платит за
    # загрузку сорок секунд сверху — на экзамене между вопросами как
    # раз такие паузы. Модель нужна всё время, пока приложение открыто.
    keep_alive: str = "2h"


class OllamaClient:
    def __init__(self, host: str, options: LLMOptions):
        self.host = host.rstrip("/")
        self.options = options
        self._session = requests.Session()
        # Обращаться к Ollama могут два потока сразу: конвейер считает
        # ответ, а прогрев тяжёлой модели идёт фоном, пока человек
        # диктует. requests.Session один пул соединений на всех, и
        # делить его без замка нельзя. Замок рекурсивный: внутри
        # генерации тот же поток вызывает выгрузку соседней модели.
        self._lock = threading.RLock()

    @classmethod
    def from_config(cls, cfg) -> "OllamaClient":
        llm_profile = cfg.profile["llm"]
        options = LLMOptions(
            model=llm_profile["model"],
            num_ctx=llm_profile.get("num_ctx", 8192),
            num_gpu=llm_profile.get("num_gpu", 0),
            temperature=cfg.get("llm.temperature", 0.2),
            top_p=cfg.get("llm.top_p", 0.9),
            timeout=cfg.get("llm.request_timeout", 180),
            stream_timeout=cfg.get("llm.stream_timeout", 300),
            keep_alive=cfg.get("llm.keep_alive", "2h"),
        )
        return cls(cfg.get("llm.host", "http://127.0.0.1:11434"), options)

    # ------------------------------------------------------------ состояние
    def ping(self, timeout: float = 3.0) -> bool:
        try:
            response = self._session.get(f"{self.host}/api/version", timeout=timeout)
            return response.ok
        except requests.RequestException:
            return False

    def available_models(self, timeout: float = 5.0) -> list[str]:
        try:
            response = self._session.get(f"{self.host}/api/tags", timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise LLMUnavailable(f"Ollama не отвечает на {self.host}: {exc}") from exc
        return [m["name"] for m in response.json().get("models", [])]

    def ensure_running(self, wait_s: float = 25.0) -> bool:
        """Поднимает Ollama, если она не отвечает.

        Без Ollama бот бесполезен целиком: не работает ни поиск по
        смыслу, ни сам ответ. При этом она легко оказывается выключенной
        — после перезагрузки, после закрытия окна, — и пользователь
        видит лишь «в базе не нашлось» вместо внятной причины. Проще
        запустить её самим, чем объяснять, почему бот молчит.

        Запускаем отдельным процессом, не привязанным к нашему окну:
        иначе Ollama умрёт вместе с ботом и следующий запуск снова
        начнётся с загрузки модели.
        """
        if self.ping():
            return True

        import shutil

        binary = shutil.which("ollama")
        if not binary:
            log.warning("Ollama не найдена в PATH — запустить нечем")
            return False

        log.info("Ollama не отвечает, запускаю: %s serve", binary)
        try:
            creation = getattr(subprocess, "CREATE_NO_WINDOW", 0) | \
                getattr(subprocess, "DETACHED_PROCESS", 0)
            subprocess.Popen([binary, "serve"], creationflags=creation,
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        except Exception:
            log.exception("Не удалось запустить Ollama")
            return False

        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            if self.ping(timeout=2.0):
                log.info("Ollama поднялась за %.0f с",
                         wait_s - (deadline - time.monotonic()))
                return True
            time.sleep(1.0)
        log.warning("Ollama не ответила за %.0f с после запуска", wait_s)
        return False

    def check_ready(self) -> tuple[bool, str]:
        """(готов, человекочитаемая причина) — для показа в оверлее."""
        if not self.ping() and not self.ensure_running():
            return False, (f"Ollama не запущена ({self.host}). "
                           "Запустите 'ollama serve' и повторите.")
        try:
            models = self.available_models()
        except LLMUnavailable as exc:
            return False, str(exc)

        wanted = self.options.model
        # Ollama хранит теги как 'qwen3:4b'; без тега подразумевается ':latest'.
        normalized = wanted if ":" in wanted else f"{wanted}:latest"
        if normalized not in models and wanted not in models:
            return False, (f"Модель {wanted} не скачана. "
                           f"Выполните: ollama pull {wanted}")
        return True, f"Ollama готова, модель {wanted}"

    # ------------------------------------------------------------ генерация
    def _payload(self, messages: list[Message], stream: bool,
                 num_predict: int = 0, model: str = "") -> dict:
        payload = {
            "model": model or self.options.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": stream,
            "keep_alive": self.options.keep_alive,
            "options": {
                "num_ctx": self.options.num_ctx,
                "num_gpu": self.options.num_gpu,
                "temperature": self.options.temperature,
                "top_p": self.options.top_p,
            },
        }
        limit = num_predict or self.options.num_predict
        if limit:
            payload["options"]["num_predict"] = limit
        if self.options.stop:
            payload["options"]["stop"] = self.options.stop
        # Параметр "think" сюда СОЗНАТЕЛЬНО не передаётся.
        # Проверено на Ollama 0.32.15 + qwen3:4b: "think": false не выключает
        # рассуждения, а ломает их разбор — модель всё равно рассуждает, но
        # текст рассуждений вываливается в content вместе с тегом </think>.
        # Без этого параметра Ollama штатно кладёт рассуждения в отдельное
        # поле message.thinking, а content остаётся чистым.
        # Директива /no_think в промпте этой версией тоже игнорируется,
        # поэтому единственный рабочий способ убрать задержку на рассуждения —
        # брать модель без режима мышления (см. README, выбор модели).
        return payload

    def stream_chat(self, messages: list[Message],
                    num_predict: int = 0,
                    model: str = "") -> Iterator[str]:
        """Отдаёт куски готового ответа по мере генерации.

        Рассуждения модели (message.thinking) в выдачу не попадают: их
        показывать пользователю не нужно. Если нужен индикатор «думает» —
        передайте on_thinking.
        """
        yield from self.stream_chat_ex(messages)

    def stream_chat_ex(self, messages: list[Message],
                       on_thinking: "Callable[[str], None] | None" = None,
                       num_predict: int = 0,
                       model: str = "",
                       timeout: int = 0) -> Iterator[str]:
        """timeout — свой бюджет времени на этот запрос, в секундах.

        Нужен для многопроходного дела: там пять обращений подряд, и
        общий потолок в пять минут означает, что один застрявший проход
        съедает всё ожидание. Каждому проходу отмеряем своё.
        """
        budget = timeout or self.options.stream_timeout
        with self._lock:
            try:
                response = self._session.post(
                    f"{self.host}/api/chat",
                    json=self._payload(messages, stream=True,
                                       num_predict=num_predict, model=model),
                    stream=True,
                    timeout=min(self.options.timeout, budget),
                )
                response.raise_for_status()
            except requests.RequestException as exc:
                raise LLMUnavailable(
                    f"Запрос к Ollama не прошёл: {exc}") from exc

        # Общий дедлайн на весь ответ. Таймаута requests тут мало:
        # при потоковом ответе Ollama сразу отдаёт заголовки, а потом
        # молча грузит модель с диска. Соединение при этом живо, и
        # запрос висит бесконечно — на 14B бот так простоял пятнадцать
        # минут, показывая «Шаг 1 из 5».
        #
        # Чтение потока обязано быть внутри try. Обрыв соединения или
        # таймаут чтения происходят уже после того, как запрос ушёл, и
        # requests бросает их прямо из iter_lines. Раньше эта ветка
        # была снаружи: наверх летел ReadTimeout вместо LLMUnavailable,
        # его никто не ловил, поток задачи умирал молча — и на экране
        # навсегда оставалось «Шаг 1 из 5».
        deadline = time.monotonic() + budget
        try:
            lines = response.iter_lines(decode_unicode=True)
            yield from self._read_stream(lines, response, deadline,
                                         on_thinking)
        except requests.RequestException as exc:
            response.close()
            raise LLMUnavailable(
                f"Связь с Ollama оборвалась на полуслове: {exc}") from exc

    def _read_stream(self, lines, response, deadline: float,
                     on_thinking: "Callable[[str], None] | None"
                     ) -> Iterator[str]:
        """Разбор потока ответа. Вынесен отдельно, чтобы обрыв связи
        внутри него ловился одним местом выше."""
        for line in lines:
            if time.monotonic() > deadline:
                response.close()
                raise LLMUnavailable(
                    f"Ollama не ответила за {budget} с — возможно, "
                    f"модель не помещается в память и грузится с диска")
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                log.warning("Не разобран фрагмент ответа Ollama: %r", line[:200])
                continue
            if chunk.get("error"):
                raise LLMUnavailable(f"Ollama вернула ошибку: {chunk['error']}")

            message = chunk.get("message", {})
            reasoning = message.get("thinking")
            if reasoning and on_thinking is not None:
                on_thinking(reasoning)

            piece = message.get("content", "")
            if piece:
                yield piece
            if chunk.get("done"):
                break

    def chat(self, messages: list[Message], num_predict: int = 0,
             model: str = "", timeout: int = 0) -> str:
        return "".join(self.stream_chat_ex(messages, num_predict=num_predict,
                                           model=model, timeout=timeout))

    def unload(self, model: str) -> None:
        """Выгружает модель из видеопамяти.

        Нужно при переключении между моделями: 14B занимает больше
        десяти гигабайт, и рядом с 7B они в шестнадцать не помещаются.
        Ollama сама выгружает по таймеру keep_alive, но ждать два часа
        посреди работы нельзя — освобождаем сразу.
        """
        if not model:
            return
        try:
            with self._lock:
                self._session.post(
                    f"{self.host}/api/chat",
                    json={"model": model, "keep_alive": 0, "messages": []},
                    timeout=30)
            log.info("Модель выгружена из памяти: %s", model)
        except requests.RequestException as exc:
            log.warning("Не удалось выгрузить %s: %s", model, exc)

    def stop_server(self) -> bool:
        """Останавливает сам процесс Ollama, а не только модель.

        Нужно для «выйти совсем»: Ollama нарочно переживает выход бота,
        чтобы следующий запуск не ждал загрузки модели, — но когда
        человек закрывает всё, она остаётся держать несколько гигабайт
        видеопамяти, и об этом никто не догадывается.
        """
        try:
            subprocess.run(["taskkill", "/F", "/IM", "ollama.exe", "/T"],
                           capture_output=True,
                           creationflags=getattr(subprocess,
                                                 "CREATE_NO_WINDOW", 0))
            # app.exe — отдельный процесс десктопного клиента Ollama;
            # без него значок ламы остаётся висеть в трее.
            subprocess.run(["taskkill", "/F", "/IM", "ollama app.exe", "/T"],
                           capture_output=True,
                           creationflags=getattr(subprocess,
                                                 "CREATE_NO_WINDOW", 0))
            log.info("Ollama остановлена")
            return True
        except Exception:
            log.exception("Не удалось остановить Ollama")
            return False

    def loaded_sizes(self) -> dict[str, int]:
        """Сколько байт видеопамяти держит каждая загруженная модель.

        Нужно, чтобы решить, влезет ли тяжёлая модель: память, занятую
        нашими же моделями, можно освободить, и считать её занятой
        неправильно.
        """
        try:
            with self._lock:
                response = self._session.get(f"{self.host}/api/ps", timeout=10)
            response.raise_for_status()
            return {m["name"]: int(m.get("size_vram") or 0)
                    for m in response.json().get("models", [])}
        except (requests.RequestException, ValueError, KeyError, TypeError):
            return {}

    def loaded_models(self) -> list[str]:
        """Какие модели сейчас держит Ollama в памяти."""
        try:
            with self._lock:
                response = self._session.get(f"{self.host}/api/ps", timeout=10)
            response.raise_for_status()
            return [m["name"] for m in response.json().get("models", [])]
        except (requests.RequestException, ValueError, KeyError):
            return []
