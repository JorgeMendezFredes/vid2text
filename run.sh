#!/bin/bash
# === run.sh ===
cd "$(dirname "$0")" || exit 1

# Activar entorno virtual
source .venv/bin/activate

# Rutas de librerías CUDA instaladas por pip (cuBLAS/cuDNN)
VENV_SITEPKG="$(python - <<'PY'
import site; print(site.getsitepackages()[0])
PY
)"
export LD_LIBRARY_PATH="$VENV_SITEPKG/nvidia/cublas/lib:$VENV_SITEPKG/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"
unset VENV_SITEPKG

# Ejecutar interfaz gráfica
python gui_transcribe_whisper.py
