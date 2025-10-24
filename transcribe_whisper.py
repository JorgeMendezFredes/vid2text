#!/usr/bin/env python3
# transcribe_whisper.py
# Uso:
#   python transcribe_whisper.py --input "/ruta/a/carpeta" --output "/ruta/a/salidas" \
#     --mode both --model small.en --device auto --vad off --overwrite false
#
# Requisitos:
#   pip install faster-whisper
#   (Opcional GPU) CUDA/cuDNN compatibles
#
# Función:
#   Recorre recursivamente la carpeta de entrada. Para cada .mp4 genera:
#     - <nombre>.txt  (transcripción en inglés)
#     - <nombre>.srt  (subtítulos para VLC)
#   Mantiene estructura de carpetas bajo --output. Omite archivos ya procesados salvo --overwrite.

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

from faster_whisper import WhisperModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch Whisper transcription (TXT/SRT) for MP4")
    p.add_argument("--input", required=True, type=Path, help="Carpeta raíz con videos .mp4")
    p.add_argument("--output", required=False, type=Path, default=None, help="Carpeta de salida (default: junto al input)")
    p.add_argument("--mode", choices=["both", "txt", "srt"], default="both", help="Tipos de salida por archivo")
    p.add_argument("--model", default="small.en", help="Modelo faster-whisper (p.ej., base.en, small.en, medium.en)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Dispositivo")
    p.add_argument("--compute-type", default="auto", help="float16, int8, int8_float16, auto")
    p.add_argument("--vad", choices=["off", "on"], default="off", help="Voice Activity Detection")
    p.add_argument("--beam-size", type=int, default=5, help="Beam size para decodificación")
    p.add_argument("--overwrite", type=str, default="false", help="true/false: regenerar salidas existentes")
    return p.parse_args()


def decide_device_and_compute(device: str, compute_type: str) -> Tuple[str, str]:
    if device == "auto":
        # Intento CUDA si existe, si no CPU
        try:
            import torch  # noqa
            if torch.cuda.is_available():  # type: ignore[attr-defined]
                dev = "cuda"
            else:
                dev = "cpu"
        except Exception:
            dev = "cpu"
    else:
        dev = device

    if compute_type != "auto":
        return dev, compute_type

    # Heurística por defecto
    if dev == "cuda":
        return dev, "float16"
    return dev, "int8"


def find_mp4_files(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*.mp4") if p.is_file()])


def rel_output_paths(input_root: Path, output_root: Path, video_path: Path) -> Tuple[Path, Path]:
    rel_dir = video_path.parent.relative_to(input_root)
    out_dir = output_root.joinpath(rel_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = video_path.stem
    txt_path = out_dir.joinpath(f"{base}.txt")
    srt_path = out_dir.joinpath(f"{base}.srt")
    return txt_path, srt_path


def s_to_srt_ts(t: float) -> str:
    if t < 0:
        t = 0.0
    hours = int(t // 3600)
    minutes = int((t % 3600) // 60)
    seconds = int(t % 60)
    millis = int(round((t - int(t)) * 1000))
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


def write_srt(segments, srt_path: Path) -> None:
    with srt_path.open("w", encoding="utf-8") as f:
        for idx, seg in enumerate(segments, start=1):
            start_ts = s_to_srt_ts(seg.start)
            end_ts = s_to_srt_ts(seg.end)
            text = seg.text.strip()
            if not text:
                continue
            f.write(f"{idx}\n{start_ts} --> {end_ts}\n{text}\n\n")


def write_txt(full_text: str, txt_path: Path) -> None:
    txt_path.write_text(full_text.strip() + "\n", encoding="utf-8")


def transcribe_file(
    model: WhisperModel,
    video_path: Path,
    language: str,
    beam_size: int,
    use_vad: bool,
) -> Tuple[str, List]:
    segments, info = model.transcribe(
        str(video_path),
        language=language,
        beam_size=beam_size,
        vad_filter=use_vad,
        vad_parameters={"min_silence_duration_ms": 500} if use_vad else None,
    )
    seg_list = list(segments)
    full_text = " ".join(seg.text.strip() for seg in seg_list if seg.text.strip())
    return full_text, seg_list


def main() -> int:
    args = parse_args()

    input_root: Path = args.input.resolve()
    if not input_root.exists() or not input_root.is_dir():
        print(f"[ERROR] Carpeta de entrada inválida: {input_root}", file=sys.stderr)
        return 2

    output_root: Path = args.output.resolve() if args.output else input_root.resolve()

    overwrite = str(args.overwrite).lower() in {"1", "true", "yes", "y"}
    use_vad = args.vad == "on"
    device, compute_type = decide_device_and_compute(args.device, args.compute_type)

    print(f"[INFO] Modelo: {args.model} | Device: {device} | Compute: {compute_type} | VAD: {use_vad}")
    print(f"[INFO] Input: {input_root}")
    print(f"[INFO] Output: {output_root}")
    print(f"[INFO] Mode: {args.mode} | Overwrite: {overwrite}")

    t0 = time.time()
    model = WhisperModel(args.model, device=device, compute_type=compute_type)

    videos = find_mp4_files(input_root)
    if not videos:
        print("[WARN] No se encontraron archivos .mp4.")
        return 0

    processed = 0
    skipped = 0
    failed = 0

    for vid in videos:
        txt_path, srt_path = rel_output_paths(input_root, output_root, vid)

        need_txt = args.mode in ("both", "txt")
        need_srt = args.mode in ("both", "srt")

        if not overwrite:
            if (not need_txt or txt_path.exists()) and (not need_srt or srt_path.exists()):
                print(f"[SKIP] {vid} (salidas existentes)")
                skipped += 1
                continue

        print(f"[RUN ] {vid}")
        start = time.time()
        try:
            full_text, segments = transcribe_file(
                model=model,
                video_path=vid,
                language="en",
                beam_size=args.beam_size,
                use_vad=use_vad,
            )
            if need_txt:
                write_txt(full_text, txt_path)
            if need_srt:
                write_srt(segments, srt_path)
            dt = time.time() - start
            print(f"[OK  ] {vid.name} | {dt:.1f}s | TXT:{need_txt} SRT:{need_srt}")
            processed += 1
        except Exception as e:
            dt = time.time() - start
            print(f"[FAIL] {vid.name} | {dt:.1f}s | {e}", file=sys.stderr)
            failed += 1

    total = time.time() - t0
    print(f"[DONE] Procesados: {processed} | Omitidos: {skipped} | Fallidos: {failed} | Tiempo total: {total:.1f}s")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
