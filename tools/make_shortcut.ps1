# Ярлык «Tobi» на рабочем столе.
#
# Запускает ассистента без консольного окна (pythonw), с иконкой-мордочкой
# из assets\tobi.ico. Скрипт лежит в проекте, чтобы ярлык можно было
# пересоздать после переустановки Python или переноса папки — путь к
# интерпретатору он находит сам.
#
# Запуск: powershell -ExecutionPolicy Bypass -File tools\make_shortcut.ps1

$ErrorActionPreference = 'Stop'

$project = Split-Path -Parent $PSScriptRoot
$icon = Join-Path $project 'assets\tobi.ico'
$entry = Join-Path $project 'main.py'

if (-not (Test-Path $entry)) { throw "Не найден main.py в $project" }
if (-not (Test-Path $icon)) { throw "Нет иконки: сначала python tools\make_icon.py" }

# pythonw вместо python: иначе за окном ассистента висит чёрная консоль.
$pythonw = (Get-Command pythonw -ErrorAction SilentlyContinue).Source
if (-not $pythonw) {
    $python = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $python) { throw 'Python не найден в PATH' }
    $pythonw = Join-Path (Split-Path -Parent $python) 'pythonw.exe'
}

$desktop = [Environment]::GetFolderPath('Desktop')
$shell = New-Object -ComObject WScript.Shell

function New-TobiShortcut {
    param($Name, $ExtraArgs, $Description)

    $path = Join-Path $desktop $Name
    $link = $shell.CreateShortcut($path)
    $link.TargetPath = $pythonw
    $link.Arguments = ("`"$entry`" " + $ExtraArgs).Trim()
    $link.WorkingDirectory = $project
    $link.IconLocation = "$icon,0"
    $link.Description = $Description
    $link.WindowStyle = 7      # свёрнутым: окно рисует сам оверлей
    $link.Save()
    Write-Output "Ярлык создан: $path"
    Write-Output "  аргументы : $($link.Arguments)"
}

New-TobiShortcut 'Tobi.lnk' '' 'Tobi — голосовой помощник по законам Eclipse RP'

# Второй ярлык — для игры на максималках. Enhanced на ультрах забирает
# 12-13 ГБ из 16, и обычный профиль в остаток не помещается: драйвер
# начинает вытеснять страницы модели, ответ растягивается на полторы
# минуты. В этом профиле бот ужат до трёх гигабайт.
New-TobiShortcut 'Tobi (игра на максималках).lnk' '--profile game_max' `
    'Tobi в облегчённом режиме: для GTA Enhanced на максимальных настройках'
