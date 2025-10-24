# Whisper Batch GUI (Pop!_OS / Ubuntu)

Aplicación por terminal y GUI para transcribir en **inglés** vídeos `.mp4` usando **faster-whisper** (CTranslate2).  
Funciona en CPU o GPU. En GPU usa librerías **CUDA/cuBLAS/cuDNN** instaladas vía `pip` dentro del `venv`.

---

## Características
- Procesamiento por lote de `.mp4`.
- Salida por archivo: **.txt** y/o **.srt** usando el mismo nombre base del video.
- Modelos: `base.en`, `small.en`, `medium.en`, `large-v3` (según VRAM/CPU).
- CLI y GUI (Tkinter).
- Soporte GPU NVIDIA opcional con wheels de NVIDIA.

---

## Requisitos
- Linux (Pop!_OS / Ubuntu 22.04+ probado).
- Python 3.10+.
- Para GUI: `python3-tk`.
- GPU NVIDIA opcional. Recomendado para `small.en` con 4 GB VRAM.

---

## Estructura
```
transcribe/
├── gui_transcribe_whisper.py
├── transcribe_whisper.py
├── run.sh
└── test/                  # tus .mp4 de prueba
```

---

## Instalación rápida

### 1) Crear entorno
```bash
cd transcribe
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Dependencias base
```bash
pip install -U faster-whisper ctranslate2
sudo apt install -y python3-tk   # GUI
```

### 3) (CPU) Ejecutar sin GPU
```bash
python transcribe_whisper.py \
  --input "/ruta/a/videos" --output "/ruta/salida" \
  --mode both --model small.en --device cpu --compute-type int8
```

### 4) (GPU) Instalar CUDA libs por `pip` dentro del venv
> Ver tutorial completo en **docs/CUDA_SETUP.md**. Resumen mínimo:
```bash
pip install -U "nvidia-cublas-cu12>=12.4,<13" "nvidia-cudnn-cu12>=9,<10"
```

Exporta rutas (temporal para la sesión):
```bash
export LD_LIBRARY_PATH="$(python - <<'PY'
import site,os
sp=site.getsitepackages()[0]
print(f"{sp}/nvidia/cublas/lib:{sp}/nvidia/cudnn/lib")
PY
):${LD_LIBRARY_PATH}"
```

Probar:
```bash
python transcribe_whisper.py \
  --input "/ruta/a/videos" --output "/ruta/salida" \
  --mode both --model small.en --device cuda --compute-type int8_float16 --overwrite true
```

---

## GUI

Lanzador que activa el venv, exporta rutas CUDA y abre la interfaz:
```bash
chmod +x run.sh
./run.sh
```

Uso:
1. Carpeta de entrada y salida.  
2. Modo: `txt`, `srt` o `both`.  
3. Modelo y opciones.  
4. Iniciar.

---

## CLI: opciones
```bash
python transcribe_whisper.py \
  --input DIR_IN \
  --output DIR_OUT \
  --mode [txt|srt|both] \
  --model [base.en|small.en|medium.en|large-v3] \
  --device [auto|cpu|cuda] \
  --compute-type [auto|float16|int8|int8_float16] \
  --vad [on|off] \
  --beam-size INT \
  --overwrite [true|false]
```

**Sugerencias:**
- 4 GB VRAM: `small.en` + `--compute-type int8_float16`.  
- Máxima fidelidad sin GPU: `medium.en` en CPU con `--compute-type int8`.

---

## Solución de problemas
- `ModuleNotFoundError: tkinter`: `sudo apt install -y python3-tk`  
- `libcublas.so.12 not found`: instala `nvidia-cublas-cu12` y exporta `LD_LIBRARY_PATH`  
- `cudnnCreateTensorDescriptor`: instala `nvidia-cudnn-cu12` y exporta `LD_LIBRARY_PATH`  
- Verifica rutas:
```bash
python - <<'PY'
import site,glob,os
sp=site.getsitepackages()[0]
print("cublas:", glob.glob(os.path.join(sp,'nvidia/cublas/lib/libcublas.so.*')))
print("cudnn :", glob.glob(os.path.join(sp,'nvidia/cudnn/lib/libcudnn*.so*')))
PY
```

---

## Licencia
MIT — ver [`LICENSE`](LICENSE)

---

# docs/CUDA_SETUP.md

## CUDA/cuBLAS/cuDNN por pip en venv (Pop!_OS / Ubuntu)

Esta guía instala cuBLAS y cuDNN **dentro del entorno virtual** con wheels oficiales de NVIDIA.  
No requiere CUDA Toolkit del sistema.

### 1) Verifica GPU y driver
```bash
nvidia-smi
```
Debe listar tu GPU y versión de driver. El “CUDA Version” es el runtime soportado por el driver.

### 2) Crea y activa el venv
```bash
cd transcribe
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Instala librerías CUDA por pip
```bash
pip install -U "nvidia-cublas-cu12>=12.4,<13" "nvidia-cudnn-cu12>=9,<10"
```

### 4) Exporta rutas de librerías
```bash
export LD_LIBRARY_PATH="$(python - <<'PY'
import site,os
sp=site.getsitepackages()[0]
print(f"{sp}/nvidia/cublas/lib:{sp}/nvidia/cudnn/lib")
PY
):${LD_LIBRARY_PATH}"
```

Para hacerlo permanente al abrir la GUI, usa `run.sh`:
```bash
#!/bin/bash
cd "$(dirname "$0")" || exit 1
source .venv/bin/activate
VENV_SITEPKG="$(python - <<'PY'
import site; print(site.getsitepackages()[0])
PY
)"
export LD_LIBRARY_PATH="$VENV_SITEPKG/nvidia/cublas/lib:$VENV_SITEPKG/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"
unset VENV_SITEPKG
python gui_transcribe_whisper.py
```

### 5) Probar con modelo pequeño
```bash
python transcribe_whisper.py \
  --input "/ruta/a/videos" --output "/ruta/salida" \
  --mode both --model small.en --device cuda --compute-type int8_float16 --overwrite true
```

### 6) Errores comunes
- **`Library libcublas.so.12 is not found`** → falta `nvidia-cublas-cu12` o no exportaste `LD_LIBRARY_PATH`  
- **`Invalid handle ... cudnnCreateTensorDescriptor`** → falta `nvidia-cudnn-cu12`  
- **OOM en 4 GB VRAM** → usa `small.en` y `--compute-type int8_float16` o CPU

### 7) Verificaciones
```bash
python - <<'PY'
import site,glob,os
sp=site.getsitepackages()[0]
print("cuBLAS:", glob.glob(os.path.join(sp,'nvidia/cublas/lib/libcublas.so.*')))
print("cuDNN :", glob.glob(os.path.join(sp,'nvidia/cudnn/lib/libcudnn*.so*')))
PY
```

---

## .gitignore
```
.venv/
__pycache__/
*.pyc
.DS_Store
*.log
```

---

## requirements.txt
```
faster-whisper==1.2.0
ctranslate2==4.6.0
nvidia-cublas-cu12>=12.4,<13
nvidia-cudnn-cu12>=9,<10
av>=11
tqdm>=4.66
onnxruntime<2,>=1.14
```

---

## LICENSE
```
MIT License

Copyright (c) 2025 Jorge Méndez

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```