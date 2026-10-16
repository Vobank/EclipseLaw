"""Загрузка и валидация config/config.yaml."""
from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "config.yaml"

# Действия, которые работают по удержанию клавиши, а не по нажатию.
PTT_ACTIONS = {"ptt_self", "ptt_other"}


class ConfigError(RuntimeError):
    """Конфиг отсутствует или противоречив."""


class Config:
    def __init__(self, raw: dict[str, Any], path: Path):
        self._raw = raw
        self.path = path
        self.root = ROOT

    # -------------------------------------------------------------- доступ
    def get(self, dotted: str, default: Any = None) -> Any:
        """cfg.get('overlay.opacity') — путь через точку."""
        node: Any = self._raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    @property
    def profile_name(self) -> str:
        return self._raw["active_profile"]

    @property
    def profile(self) -> dict[str, Any]:
        return self._raw["profiles"][self.profile_name]

    @property
    def hotkeys(self) -> dict[str, str]:
        return dict(self._raw.get("hotkeys", {}))

    def resolve_path(self, dotted: str) -> Path:
        """Пути в конфиге относительные — разворачиваем от корня проекта."""
        value = self.get(dotted)
        if value is None:
            raise ConfigError(f"в конфиге нет пути {dotted!r}")
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    # ------------------------------------------------------------ загрузка
    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        path = path or CONFIG_PATH
        if not path.exists():
            raise ConfigError(f"не найден файл настроек: {path}")
        with path.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

        cfg = cls(raw, path)
        cfg._validate()
        cfg._apply_gpu_pinning()
        return cfg

    def _validate(self) -> None:
        profiles = self._raw.get("profiles")
        if not isinstance(profiles, dict) or not profiles:
            raise ConfigError("в конфиге нет ни одного профиля (секция profiles)")

        active = self._raw.get("active_profile")
        if active not in profiles:
            raise ConfigError(
                f"active_profile={active!r} не найден среди профилей: "
                f"{', '.join(profiles)}"
            )

        for name, profile in profiles.items():
            for section in ("stt", "llm"):
                if section not in profile:
                    raise ConfigError(f"в профиле {name!r} нет секции {section!r}")

        missing = [a for a in PTT_ACTIONS if a not in self.hotkeys]
        if missing:
            raise ConfigError(
                "в секции hotkeys не заданы обязательные PTT-клавиши: "
                + ", ".join(missing)
            )

    def _apply_gpu_pinning(self) -> None:
        """Пиннинг AI-стека на выделенную карту (ТЗ 3, этап 7.3).

        Переменную выставляем до того, как будут импортированы CUDA-рантаймы:
        ctranslate2 и llama.cpp читают её один раз при инициализации.
        Ollama — отдельный серверный процесс, его карта задаётся переменной
        окружения самой службы; здесь мы влияем только на свой процесс.
        """
        devices = self.profile.get("llm", {}).get("cuda_visible_devices")
        if devices is None:
            return
        os.environ["CUDA_VISIBLE_DEVICES"] = str(devices)
        log.info("CUDA_VISIBLE_DEVICES=%s (профиль %s)", devices, self.profile_name)


def setup_logging(cfg: Config) -> None:
    level = getattr(logging, str(cfg.get("logging.level", "INFO")).upper(), logging.INFO)
    log_file = cfg.resolve_path("logging.file")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)-22s %(message)s", "%H:%M:%S")

    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(fmt)

    console = logging.StreamHandler()
    console.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(console)
