# PowerShell installer for Phone Guard dependencies and YOLO model
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# Install CPU-only PyTorch wheels (adjust if you need CUDA)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# Download the YOLOv8n model weights
python download_model.py
Write-Host "Installation complete. Run 'python app.py' to start." 
