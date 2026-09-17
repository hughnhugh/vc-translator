#Requires -Version 5.1
<#
    One-time setup for vc-translator: locates a usable Python install, creates
    the project's venv (venv/) if it doesn't already exist, and installs every
    dependency. Safe to re-run - an existing venv is reused, not recreated, so
    re-running this after a `git pull` just catches up on any new packages.

    This does NOT install Python or NVIDIA drivers themselves - those are
    system-wide changes this script won't make for you; it just detects them
    and tells you what's missing.
#>

$ErrorActionPreference = "Stop"

function Write-Step($text) {
    Write-Host ""
    Write-Host "==> $text" -ForegroundColor Cyan
}

function Write-Warn($text) {
    Write-Host "WARNING: $text" -ForegroundColor Yellow
}

function Exit-WithError($text) {
    Write-Host ""
    Write-Host "ERROR: $text" -ForegroundColor Red
    exit 1
}

# --- Locate Python -----------------------------------------------------
Write-Step "Looking for Python 3.12"

$pythonExe = $null
$pythonPrefixArgs = @()

try {
    $pyVersion = & py -3.12 --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        $pythonExe = "py"
        $pythonPrefixArgs = @("-3.12")
        Write-Host "Found via 'py' launcher: $pyVersion"
    }
} catch {}

if (-not $pythonExe) {
    try {
        $pyVersion = & python --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            if ($pyVersion -notmatch "3\.12") {
                Write-Warn "Found $pyVersion, not 3.12 - vc-translator is tested on 3.12. Continuing anyway; if package installs fail below, install Python 3.12 from https://www.python.org/downloads/ and re-run this script."
            } else {
                Write-Host "Found: $pyVersion"
            }
            $pythonExe = "python"
        }
    } catch {}
}

if (-not $pythonExe) {
    Exit-WithError "No Python install found. Install Python 3.12 from https://www.python.org/downloads/ (check 'Add python.exe to PATH' during install), then re-run this script."
}

# --- Check for an NVIDIA GPU (warn, don't block) ------------------------
Write-Step "Checking for an NVIDIA GPU"

try {
    $null = & nvidia-smi 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "NVIDIA GPU detected."
    } else {
        Write-Warn "'nvidia-smi' ran but returned an error - vc-translator requires an NVIDIA GPU with CUDA support. It will not run without one."
    }
} catch {
    Write-Warn "'nvidia-smi' not found - vc-translator requires an NVIDIA GPU with CUDA support (and its driver installed). It will not run without one."
}

# --- Create the venv (if it doesn't already exist) ----------------------
$venvPython = ".\venv\Scripts\python.exe"

if (Test-Path $venvPython) {
    Write-Step "Reusing existing venv\"
} else {
    Write-Step "Creating venv\"
    & $pythonExe @pythonPrefixArgs -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Exit-WithError "Failed to create the venv."
    }
}

function Install-Packages($description, $pipArgs, $required = $true) {
    Write-Step $description
    & $venvPython -m pip install @pipArgs
    if ($LASTEXITCODE -ne 0) {
        if ($required) {
            Exit-WithError "Failed installing: $description"
        } else {
            Write-Warn "Failed installing (optional, continuing): $description"
        }
    }
}

Install-Packages "Upgrading pip" @("--upgrade", "pip")
Install-Packages "Installing PyTorch (CUDA build - this is a large download)" `
    @("torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu128")
Install-Packages "Installing core dependencies" `
    @("faster-whisper", "PyAudioWPatch", "numpy", "opencc-python-reimplemented", "transformers", "sentencepiece", "websockets")
Install-Packages "Installing GPU memory readout support (optional)" @("nvidia-ml-py") -required $false

Write-Step "Setup complete"
Write-Host "Run vc-translator with:"
Write-Host "  .\venv\Scripts\python.exe translate_vc.py"
Write-Host ""
Write-Host "First launch downloads the Whisper, NLLB, and VAD models (~5GB total) -"
Write-Host "this only happens once and needs an internet connection."
