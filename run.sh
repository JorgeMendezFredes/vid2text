#!/bin/bash
# === run.sh ===
cd "$(dirname "$0")" || exit 1

# Activar entorno virtual
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "[ERROR] No se encontró .venv/bin/activate. ¿Has creado el entorno virtual?"
    exit 1
fi

# Detectar python (preferir el del venv)
PYTHON_EXE=$(which python3 || which python)

# Rutas de librerías CUDA instaladas por pip (cuBLAS/cuDNN)
VENV_SITEPKG="$($PYTHON_EXE - <<'PY'
import site; print(site.getsitepackages()[0])
PY
)"
export LD_LIBRARY_PATH="$VENV_SITEPKG/nvidia/cublas/lib:$VENV_SITEPKG/nvidia/cudnn/lib:$VENV_SITEPKG/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH}"
unset VENV_SITEPKG

# Ejecutar interfaz gráfica
$PYTHON_EXE gui_transcribe_whisper.py
