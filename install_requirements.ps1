param(
	[switch]$UseCuda,
	[switch]$SkipModel
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Write-Host "Starting installation (UseCuda=$UseCuda, SkipModel=$SkipModel)"

try {
	$python = (Get-Command python -ErrorAction Stop).Source
} catch {
	Write-Error "Python not found in PATH. Please install Python 3.8+ and add it to PATH."
	exit 1
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$venv = Join-Path $root 'venv'

if (-not (Test-Path $venv)) {
	Write-Host "Creating virtual environment at $venv"
	& $python -m venv $venv
}

$venv_py = Join-Path $venv 'Scripts\python.exe'
if (-not (Test-Path $venv_py)) {
	Write-Warning "Virtualenv python not found; falling back to system python."
	$venv_py = $python
}

Write-Host "Upgrading pip and installing required Python packages..."
& $venv_py -m pip install --upgrade pip
& $venv_py -m pip install -r (Join-Path $root 'requirements.txt')

if ($UseCuda) {
	Write-Host "Installing CUDA-enabled PyTorch (ensure drivers/CUDA are compatible)..."
	& $venv_py -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
} else {
	Write-Host "Installing CPU PyTorch wheels..."
	& $venv_py -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
}

if (-not $SkipModel) {
	Write-Host "Downloading YOLO model weights..."
	& $venv_py (Join-Path $root 'download_model.py')
}

Write-Host "Installation complete." -ForegroundColor Green
Write-Host "To activate the venv: .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "Then run: python app.py"

exit 0
