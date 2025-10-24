#!/usr/bin/env python3
# gui_transcribe_whisper.py
# GUI cross-platform que llama al script transcribe_whisper.py por debajo.
# Requisitos: Python 3.10+, Tkinter (incluido), y el archivo transcribe_whisper.py en la MISMA carpeta.

import sys
import os
import threading
import subprocess
import queue
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

APP_TITLE = "Whisper Batch GUI"
HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "transcribe_whisper.py"

class WhisperGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("760x560")
        self.minsize(720, 520)

        if not SCRIPT.exists():
            messagebox.showerror("Error", f"No se encontró {SCRIPT.name} en {HERE}")
            self.destroy()
            return

        self.proc = None
        self.reader_thread = None
        self.q = queue.Queue()
        self._build_ui()
        self._poll_queue()

    def _build_ui(self):
        pad = {"padx": 8, "pady": 6}

        frm = ttk.Frame(self)
        frm.pack(fill="x", **pad)

        # Input dir
        ttk.Label(frm, text="Carpeta de entrada (.mp4):").grid(row=0, column=0, sticky="w")
        self.var_input = tk.StringVar()
        ent_input = ttk.Entry(frm, textvariable=self.var_input, width=70)
        ent_input.grid(row=0, column=1, sticky="ew", **pad)
        btn_in = ttk.Button(frm, text="Elegir…", command=self._choose_input)
        btn_in.grid(row=0, column=2, **pad)

        # Output dir
        ttk.Label(frm, text="Carpeta de salida:").grid(row=1, column=0, sticky="w")
        self.var_output = tk.StringVar()
        ent_output = ttk.Entry(frm, textvariable=self.var_output, width=70)
        ent_output.grid(row=1, column=1, sticky="ew", **pad)
        btn_out = ttk.Button(frm, text="Elegir…", command=self._choose_output)
        btn_out.grid(row=1, column=2, **pad)

        frm.columnconfigure(1, weight=1)

        # Options frame
        opt = ttk.LabelFrame(self, text="Opciones")
        opt.pack(fill="x", **pad)

        ttk.Label(opt, text="Modo:").grid(row=0, column=0, sticky="w")
        self.var_mode = tk.StringVar(value="both")
        cmb_mode = ttk.Combobox(opt, textvariable=self.var_mode, state="readonly",
                                values=["both", "txt", "srt"], width=10)
        cmb_mode.grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(opt, text="Modelo:").grid(row=0, column=2, sticky="w")
        self.var_model = tk.StringVar(value="small.en")
        cmb_model = ttk.Combobox(opt, textvariable=self.var_model, state="readonly",
                                 values=["base.en", "small.en", "medium.en", "large-v3"], width=12)
        cmb_model.grid(row=0, column=3, sticky="w", **pad)

        ttk.Label(opt, text="Device:").grid(row=1, column=0, sticky="w")
        self.var_device = tk.StringVar(value="auto")
        cmb_device = ttk.Combobox(opt, textvariable=self.var_device, state="readonly",
                                  values=["auto", "cpu", "cuda"], width=10)
        cmb_device.grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(opt, text="Compute:").grid(row=1, column=2, sticky="w")
        self.var_compute = tk.StringVar(value="auto")
        cmb_compute = ttk.Combobox(opt, textvariable=self.var_compute, state="readonly",
                                   values=["auto", "float16", "int8", "int8_float16"], width=12)
        cmb_compute.grid(row=1, column=3, sticky="w", **pad)

        ttk.Label(opt, text="Beam size:").grid(row=2, column=0, sticky="w")
        self.var_beam = tk.IntVar(value=5)
        spn_beam = ttk.Spinbox(opt, from_=1, to=10, textvariable=self.var_beam, width=5)
        spn_beam.grid(row=2, column=1, sticky="w", **pad)

        self.var_vad = tk.BooleanVar(value=False)
        chk_vad = ttk.Checkbutton(opt, text="VAD", variable=self.var_vad)
        chk_vad.grid(row=2, column=2, sticky="w", **pad)

        self.var_overwrite = tk.BooleanVar(value=False)
        chk_over = ttk.Checkbutton(opt, text="Overwrite", variable=self.var_overwrite)
        chk_over.grid(row=2, column=3, sticky="w", **pad)

        for c in range(4):
            opt.columnconfigure(c, weight=1)

        # Buttons
        act = ttk.Frame(self)
        act.pack(fill="x", **pad)
        self.btn_run = ttk.Button(act, text="Iniciar", command=self._on_run)
        self.btn_run.pack(side="left", padx=4)
        self.btn_stop = ttk.Button(act, text="Detener", command=self._on_stop, state="disabled")
        self.btn_stop.pack(side="left", padx=4)
        ttk.Button(act, text="Abrir salida", command=self._open_output).pack(side="left", padx=4)

        # Progress
        self.prog = ttk.Progressbar(self, mode="indeterminate")
        self.prog.pack(fill="x", **pad)

        # Log
        logf = ttk.LabelFrame(self, text="Registro")
        logf.pack(fill="both", expand=True, **pad)
        self.txt = tk.Text(logf, wrap="word", height=18)
        self.txt.pack(fill="both", expand=True, padx=6, pady=6)
        self._log(f"{APP_TITLE} listo.\nScript: {SCRIPT}\nPython: {sys.executable}\n")

    def _choose_input(self):
        d = filedialog.askdirectory(title="Seleccionar carpeta de entrada")
        if d:
            self.var_input.set(d)
            if not self.var_output.get():
                self.var_output.set(d)

    def _choose_output(self):
        d = filedialog.askdirectory(title="Seleccionar carpeta de salida")
        if d:
            self.var_output.set(d)

    def _open_output(self):
        path = self.var_output.get().strip()
        if not path:
            messagebox.showwarning("Aviso", "Define la carpeta de salida.")
            return
        p = Path(path)
        if not p.exists():
            messagebox.showwarning("Aviso", "La carpeta de salida no existe.")
            return
        if sys.platform.startswith("win"):
            os.startfile(str(p))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(p)])
        else:
            subprocess.Popen(["xdg-open", str(p)])

    def _on_run(self):
        if self.proc is not None:
            messagebox.showwarning("En ejecución", "Ya hay un proceso en curso.")
            return
        input_dir = self.var_input.get().strip()
        output_dir = self.var_output.get().strip() or input_dir
        if not input_dir:
            messagebox.showwarning("Falta entrada", "Selecciona la carpeta de entrada.")
            return
        if not Path(input_dir).exists():
            messagebox.showerror("Error", "La carpeta de entrada no existe.")
            return

        cmd = [
            sys.executable, str(SCRIPT),
            "--input", input_dir,
            "--output", output_dir,
            "--mode", self.var_mode.get(),
            "--model", self.var_model.get(),
            "--device", self.var_device.get(),
            "--compute-type", self.var_compute.get(),
            "--vad", "on" if self.var_vad.get() else "off",
            "--beam-size", str(self.var_beam.get()),
            "--overwrite", "true" if self.var_overwrite.get() else "false",
        ]
        self._log(f"$ {' '.join(self._quote(a) for a in cmd)}\n")

        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as e:
            self._log(f"[ERROR] No se pudo iniciar el proceso: {e}\n")
            self.proc = None
            return

        self.btn_run.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.prog.start(12)
        self.reader_thread = threading.Thread(target=self._reader, daemon=True)
        self.reader_thread.start()

    def _on_stop(self):
        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass

    def _reader(self):
        assert self.proc is not None
        with self.proc.stdout:
            for line in self.proc.stdout:
                self.q.put(line)
        code = self.proc.wait()
        self.q.put(f"\n[EXIT] Código: {code}\n")
        self.q.put("__PROC_DONE__")

    def _poll_queue(self):
        try:
            while True:
                msg = self.q.get_nowait()
                if msg == "__PROC_DONE__":
                    self._on_done()
                else:
                    self._log(msg)
        except queue.Empty:
            pass
        self.after(60, self._poll_queue)

    def _on_done(self):
        self.prog.stop()
        self.btn_run.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.proc = None
        self.reader_thread = None

    def _log(self, s: str):
        self.txt.insert("end", s)
        self.txt.see("end")

    @staticmethod
    def _quote(a: str) -> str:
        if " " in a or "\t" in a:
            if sys.platform.startswith("win"):
                return f'"{a}"'
            return f"'{a}'"
        return a

if __name__ == "__main__":
    app = WhisperGUI()
    if app.winfo_exists():
        app.mainloop()
