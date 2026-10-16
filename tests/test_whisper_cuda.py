"""
Подключение библиотек CUDA к Whisper.

Файл появился после сбоя, который выглядел как неисправность карты:
модель загружалась, в логе стояло «Библиотеки CUDA подключены», а на
первом же вопросе распознавание падало с «cublas64_12.dll is not found».

Причин оказалось две, и обе про пути к библиотекам.

Первая: os.add_dll_directory действует только на загрузку через Python,
а cuBLAS подтягивает CTranslate2 из C++ обычным LoadLibrary — тот
смотрит стандартные пути, включая PATH. Каталоги надо добавлять в оба
места.

Вторая: add_dll_directory возвращает объект-ключ, и при его сборке
мусорщиком каталог убирается из путей обратно. Ключи надо держать.
"""
import gc
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.stt import whisper_engine  # noqa: E402
from core.stt.whisper_engine import WhisperEngine  # noqa: E402


def test_folders_go_into_path(monkeypatch, tmp_path):
    """Каталоги обязаны попасть именно в PATH.

    add_dll_directory действует только на загрузку через Python.
    А cuBLAS подтягивает CTranslate2 из C++ обычным LoadLibrary, и тот
    смотрит стандартные пути, включая PATH. Без этого модель
    загружалась, в логе стояло «Библиотеки CUDA подключены», а на
    первом же вопросе приходило «cublas64_12.dll is not found».
    """
    folder = tmp_path / "nvidia" / "cublas" / "bin"
    folder.mkdir(parents=True)
    (folder / "cublas64_12.dll").write_bytes(b"")

    monkeypatch.setattr("site.getsitepackages", lambda: [str(tmp_path)])
    monkeypatch.setattr("site.getusersitepackages", lambda: str(tmp_path))
    monkeypatch.setattr(whisper_engine, "_DLL_HANDLES", [])
    monkeypatch.setenv("PATH", r"C:\Windows")
    monkeypatch.setattr("os.add_dll_directory", lambda p: object())

    assert WhisperEngine._add_cuda_libraries() == 1
    assert str(folder) in os.environ["PATH"]


def test_path_is_not_duplicated(monkeypatch, tmp_path):
    """Повторный вызов не должен раздувать PATH."""
    folder = tmp_path / "nvidia" / "cublas" / "bin"
    folder.mkdir(parents=True)
    (folder / "cublas64_12.dll").write_bytes(b"")

    monkeypatch.setattr("site.getsitepackages", lambda: [str(tmp_path)])
    monkeypatch.setattr("site.getusersitepackages", lambda: str(tmp_path))
    monkeypatch.setattr(whisper_engine, "_DLL_HANDLES", [])
    monkeypatch.setenv("PATH", r"C:\Windows")
    monkeypatch.setattr("os.add_dll_directory", lambda p: object())

    WhisperEngine._add_cuda_libraries()
    WhisperEngine._add_cuda_libraries()
    assert os.environ["PATH"].count(str(folder)) == 1


def test_handles_are_kept(monkeypatch, tmp_path):
    """Ключи каталогов обязаны пережить сборку мусора."""
    folder = tmp_path / "nvidia" / "cublas" / "bin"
    folder.mkdir(parents=True)
    (folder / "cublas64_12.dll").write_bytes(b"")

    monkeypatch.setattr("site.getsitepackages", lambda: [str(tmp_path)])
    monkeypatch.setattr("site.getusersitepackages", lambda: str(tmp_path))
    monkeypatch.setattr(whisper_engine, "_DLL_HANDLES", [])

    closed: list[str] = []

    class Handle:
        def __init__(self, path):
            self.path = path

        def __del__(self):        # так Windows и снимает каталог с поиска
            closed.append(self.path)

    monkeypatch.setattr("os.add_dll_directory", lambda p: Handle(p))

    assert WhisperEngine._add_cuda_libraries() == 1
    gc.collect()
    assert closed == [], "ключ собран мусорщиком — каталог потерян"
    assert len(whisper_engine._DLL_HANDLES) == 1


def test_missing_packages_are_reported_not_crashed(monkeypatch, tmp_path):
    """Пакетов нет — это повод остаться на CPU, а не упасть."""
    monkeypatch.setattr("site.getsitepackages", lambda: [str(tmp_path)])
    monkeypatch.setattr("site.getusersitepackages", lambda: str(tmp_path))
    monkeypatch.setattr(whisper_engine, "_DLL_HANDLES", [])
    assert WhisperEngine._add_cuda_libraries() == 0


def test_each_folder_added_once(monkeypatch, tmp_path):
    """В каталоге десять DLL — каталог всё равно один."""
    folder = tmp_path / "nvidia" / "cudnn" / "bin"
    folder.mkdir(parents=True)
    for index in range(5):
        (folder / f"lib{index}.dll").write_bytes(b"")

    monkeypatch.setattr("site.getsitepackages", lambda: [str(tmp_path)])
    monkeypatch.setattr("site.getusersitepackages", lambda: str(tmp_path))
    monkeypatch.setattr(whisper_engine, "_DLL_HANDLES", [])
    monkeypatch.setattr("os.add_dll_directory", lambda p: object())

    assert WhisperEngine._add_cuda_libraries() == 1


def test_beam_size_comes_from_the_profile():
    """На GPU широкий луч бесплатен и должен доезжать из профиля."""
    class Cfg:
        profile = {"stt": {"model": "medium", "device": "cuda",
                           "beam_size": 5}}

        def get(self, path, default=None):
            return default

    assert WhisperEngine.from_config(Cfg()).beam_size == 5
