# Sets up everything needed to run the voice agent: Ollama + model, Kokoro TTS
# model files, and a Python venv with all dependencies.
#
# Usage (from repo root):
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

Write-Host "==> Checking for Ollama..."
$ollamaExe = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
if (-not (Test-Path $ollamaExe)) {
    Write-Host "Ollama not found, installing via winget..."
    winget install --id Ollama.Ollama --source winget -e --accept-package-agreements --accept-source-agreements
} else {
    Write-Host "Ollama already installed."
}

Write-Host "==> Pulling qwen3:4b-instruct-2507-q4_K_M (~2.5GB, non-thinking variant)..."
& $ollamaExe pull qwen3:4b-instruct-2507-q4_K_M

Write-Host "==> Downloading Kokoro TTS model files (~115MB)..."
$modelsDir = Join-Path $repoRoot "models"
New-Item -ItemType Directory -Force -Path $modelsDir | Out-Null
$kokoroModel = Join-Path $modelsDir "kokoro-v1.0.int8.onnx"
$kokoroVoices = Join-Path $modelsDir "voices-v1.0.bin"
if (-not (Test-Path $kokoroModel)) {
    Invoke-WebRequest -Uri "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx" -OutFile $kokoroModel
}
if (-not (Test-Path $kokoroVoices)) {
    Invoke-WebRequest -Uri "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin" -OutFile $kokoroVoices
}

Write-Host "==> Creating Python virtual environment..."
$venvDir = Join-Path $repoRoot ".venv"
if (-not (Test-Path $venvDir)) {
    python -m venv $venvDir
}

Write-Host "==> Installing Python dependencies..."
& "$venvDir\Scripts\python.exe" -m pip install --upgrade pip -q
& "$venvDir\Scripts\pip.exe" install -r (Join-Path $repoRoot "requirements.txt") -q

Write-Host ""
Write-Host "Setup complete. Try it:"
Write-Host "  cd src"
Write-Host "  ..\.venv\Scripts\python.exe main.py --input ..\samples\sample_question_capital_of_france.wav"
Write-Host "or the web demo:"
Write-Host "  ..\.venv\Scripts\python.exe webapp.py   (then open http://127.0.0.1:5000)"
