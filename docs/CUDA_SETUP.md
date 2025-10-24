# CUDA/cuBLAS/cuDNN por pip en venv (Pop!_OS / Ubuntu)

Esta guía instala cuBLAS y cuDNN **dentro del entorno virtual** con wheels oficiales de NVIDIA.  
No requiere CUDA Toolkit del sistema.

---

## 1) Verifica GPU y driver
```bash
nvidia-smi
```

Debe listar tu GPU y versión de driver.  
El campo “CUDA Version” muestra el runtime soportado por el driver.

---

## 2) Crea y activa el venv
```bash
cd transcribe
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3) Instala librerías CUDA por pip
Instala las variantes **CUDA 12**, recomendadas para `faster-whisper` / `ctranslate2`:
```bash
pip install -U "nvidia-cublas-cu12>=12.4,<13" "nvidia-cudnn-cu12>=9,<10"
```

---

## 4) Exporta rutas de librerías
```bash
export LD_LIBRARY_PATH="$(python - <<'PY'
import site,os
sp=site.getsitepackages()[0]
print(f"{sp}/nvidia/cublas/lib:{sp}/nvidia/cudnn/lib")
PY
):${LD_LIBRARY_PATH}"
```

Para automatizarlo, el script `run.sh` ya incluye este bloque.

---

## 5) Prueba con modelo pequeño
```bash
python transcribe_whisper.py \
  --input "/ruta/a/videos" --output "/ruta/salida" \
  --mode both --model small.en --device cuda \
  --compute-type int8_float16 --overwrite true
```

---

## 6) Errores comunes

- **`Library libcublas.so.12 is not found`**  
  Falta `nvidia-cublas-cu12` o no exportaste `LD_LIBRARY_PATH`.

- **`Invalid handle ... cudnnCreateTensorDescriptor`**  
  Falta `nvidia-cudnn-cu12` o las rutas están incompletas.

- **OOM en 4 GB VRAM**  
  Usa `small.en` con `--compute-type int8_float16`.  
  Si persiste, cambia a CPU (`--device cpu`).

---

## 7) Verificaciones
```bash
python - <<'PY'
import site,glob,os
sp=site.getsitepackages()[0]
print("cuBLAS:", glob.glob(os.path.join(sp,'nvidia/cublas/lib/libcublas.so.*')))
print("cuDNN :", glob.glob(os.path.join(sp,'nvidia/cudnn/lib/libcudnn*.so*')))
PY
```
