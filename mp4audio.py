"""MP4 音訊工具：AI 去背景音 / 去人聲 + 聲場調整。

需要：ffmpeg / ffprobe / ffplay 在 PATH；pip install demucs torchaudio soundfile
"""
import os
import queue
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
SR = 44100

SEP_MODES = [
    ("不處理（只調聲場）", "none"),
    ("AI 保留人聲（去背景音樂／雜音）", "vocals"),
    ("AI 去人聲（只留背景音／伴奏）", "instrumental"),
    ("快速去人聲（中置消除，免 AI）", "center_cut"),
]

# 聲場預設：width%, surround, reverb, balance, bass, treble, voice, volume
PRESETS = {
    "原始": dict(width=100, surround=0, reverb=0, balance=0, bass=0, treble=0, voice=0, volume=0),
    "寬廣立體聲": dict(width=170, surround=20, reverb=0, balance=0, bass=1, treble=1, voice=0, volume=0),
    "虛擬環繞": dict(width=150, surround=60, reverb=15, balance=0, bass=2, treble=1, voice=0, volume=0),
    "演唱會廳": dict(width=140, surround=40, reverb=55, balance=0, bass=3, treble=0, voice=0, volume=0),
    "小房間": dict(width=90, surround=0, reverb=20, balance=0, bass=0, treble=-1, voice=0, volume=0),
    "人聲清晰": dict(width=80, surround=0, reverb=0, balance=0, bass=-2, treble=1, voice=5, volume=1),
    "重低音": dict(width=120, surround=10, reverb=0, balance=0, bass=8, treble=1, voice=0, volume=-1),
    "單聲道": dict(width=0, surround=0, reverb=0, balance=0, bass=0, treble=0, voice=0, volume=0),
}


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, creationflags=CREATE_NO_WINDOW, **kw)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode("utf-8", "replace")[-1500:])
    return r


def probe_duration(path):
    r = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path])
    try:
        return float(r.stdout.decode().strip())
    except ValueError:
        return 0.0


def has_audio(path):
    r = run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
             "stream=index", "-of", "csv=p=0", path])
    return bool(r.stdout.strip())


def has_video(path):
    """有真正的影像軌才算（MP3 內嵌的封面圖不算）。"""
    r = run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
             "stream_disposition=attached_pic", "-of", "csv=p=0", path])
    return any(line.strip() == "0" for line in r.stdout.decode().splitlines())


def build_filter(p, sep_mode, denoise):
    """依參數組 ffmpeg -af 濾鏡字串。"""
    f = []
    if sep_mode == "center_cut":
        # 左減右消掉置中的人聲，再補回低頻（低音通常也在中間）
        f.append("asplit=2[a][b];[a]pan=stereo|c0=c0-c1|c1=c1-c0[s];"
                 "[b]lowpass=f=150,pan=stereo|c0=0.5*c0+0.5*c1|c1=0.5*c0+0.5*c1[l];"
                 "[s][l]amix=inputs=2:normalize=0")
    if denoise:
        f.append("afftdn=nf=-25")
    w = p["width"] / 100.0
    if abs(w - 1.0) > 1e-3:
        f.append(f"extrastereo=m={w:.2f}:c=0")
    s = p["surround"] / 100.0
    if s > 0:
        f.append(f"stereowiden=delay={8 + 17 * s:.1f}:feedback={0.1 + 0.4 * s:.2f}"
                 f":crossfeed={0.1 + 0.3 * s:.2f}:drymix={1 - 0.3 * s:.2f}")
    r = p["reverb"] / 100.0
    if r > 0:
        f.append(f"aecho=in_gain=0.85:out_gain=0.9:delays=45|83|131|197"
                 f":decays={0.45 * r:.2f}|{0.33 * r:.2f}|{0.22 * r:.2f}|{0.12 * r:.2f}")
    if p["balance"]:
        f.append(f"stereotools=balance_out={p['balance'] / 100.0:.2f}")
    if p["bass"]:
        f.append(f"bass=g={p['bass']}:f=100")
    if p["treble"]:
        f.append(f"treble=g={p['treble']}:f=6000")
    if p["voice"]:
        f.append(f"equalizer=f=2500:t=o:w=1.5:g={p['voice']}")
    if p.get("loudnorm"):
        # 自動響度標準化到 -14 LUFS（YouTube/串流平台標準），小聲段落會被拉起來
        f.append("loudnorm=I=-14:TP=-1.5:LRA=11,aresample=44100")
    if p["volume"]:
        f.append(f"volume={p['volume']}dB")
    f.append("alimiter=limit=0.97:level=0")
    return ",".join(f)


# ---------------- AI 分離 ----------------
_model = None


def get_model():
    global _model
    if _model is None:
        import torch
        from demucs.pretrained import get_model as gm
        _model = gm("htdemucs")
        _model.eval()
        _model.to("cuda" if torch.cuda.is_available() else "cpu")
    return _model


def separate(wav_in, wav_out, mode, keep_ratio, progress):
    """wav_in: 44.1k 立體聲；mode=vocals 保留人聲 / instrumental 去人聲。
    keep_ratio: 被去掉那部分要保留多少（0~1）。progress(0~1)。"""
    import soundfile as sf
    import torch
    import demucs.apply as dapply

    model = get_model()
    device = next(model.parameters()).device
    audio, sr = sf.read(wav_in, dtype="float32", always_2d=True)
    audio = audio.T  # (2, n)
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    n = audio.shape[1]
    vi = model.sources.index("vocals")

    block = SR * 300          # 每 5 分鐘一塊，避免長片吃光記憶體
    fade = SR * 2             # 塊與塊之間 2 秒交叉淡化
    out = np.zeros_like(audio)
    weight = np.zeros(n, dtype=np.float32)
    starts = list(range(0, n, block)) or [0]
    total = len(starts)

    for bi, st in enumerate(starts):
        a = max(0, st - fade)
        b = min(n, st + block + fade)
        seg = torch.from_numpy(audio[:, a:b].copy())
        ref = seg.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        x = ((seg - mean) / std)[None].to(device)

        class _T:  # 攔截 demucs 內部 tqdm 來回報進度
            @staticmethod
            def tqdm(it, **kw):
                it = list(it)
                for i, v in enumerate(it):
                    progress((bi + i / max(1, len(it))) / total)
                    yield v
        dapply.tqdm = _T
        with torch.no_grad():
            src = dapply.apply_model(model, x, split=True, overlap=0.25,
                                     progress=True, device=device)[0]
        src = (src * std + mean).cpu().numpy()  # (4, 2, len)
        vocals = src[vi]
        rest = src.sum(0) - vocals
        if mode == "vocals":
            y = vocals + keep_ratio * rest
        else:
            y = rest + keep_ratio * vocals

        w = np.ones(b - a, dtype=np.float32)
        if a > 0:
            w[:min(2 * fade, b - a)] = np.linspace(0, 1, min(2 * fade, b - a))
        if b < n:
            k = min(2 * fade, b - a)
            w[-k:] = np.minimum(w[-k:], np.linspace(1, 0, k))
        out[:, a:b] += y * w
        weight[a:b] += w
        progress((bi + 1) / total)

    out /= np.maximum(weight, 1e-6)
    sf.write(wav_out, out.T, SR, subtype="FLOAT")


# ---------------- 主流程 ----------------
def process(src, dst, sep_mode, keep_ratio, params, denoise, log, progress,
            start=None, duration=None, audio_only=False):
    tmp = tempfile.mkdtemp(prefix="mp4audio_")
    raw = os.path.join(tmp, "raw.wav")
    sep = os.path.join(tmp, "sep.wav")
    trim = []
    if start is not None:
        trim += ["-ss", f"{start:.2f}"]
    if duration is not None:
        trim += ["-t", f"{duration:.2f}"]

    log("擷取音軌…")
    run(["ffmpeg", "-y", *trim, "-i", src, "-vn", "-ac", "2", "-ar", str(SR),
         "-c:a", "pcm_f32le", raw])
    progress(0.05)

    audio_src = raw
    if sep_mode in ("vocals", "instrumental"):
        log("AI 分離中（第一次會比較久）…")
        separate(raw, sep, sep_mode, keep_ratio,
                 lambda x: progress(0.05 + 0.8 * x))
        audio_src = sep
    progress(0.86)

    af = build_filter(params, sep_mode, denoise)
    log("套用聲場並輸出…")
    if audio_only:
        if dst.lower().endswith(".mp3"):
            codec = ["-c:a", "libmp3lame", "-b:a", "320k"]
        else:
            codec = ["-c:a", "pcm_s16le"]
        run(["ffmpeg", "-y", "-i", audio_src, "-af", af, *codec, dst])
    else:
        run(["ffmpeg", "-y", *trim, "-i", src, "-i", audio_src,
             "-filter_complex", f"[1:a]{af}[aout]",
             "-map", "0:v?", "-map", "[aout]", "-map", "0:s?",
             "-c:v", "copy", "-c:s", "copy", "-c:a", "aac", "-b:a", "256k",
             "-movflags", "+faststart", "-shortest", dst])
    progress(1.0)
    for f in (raw, sep):
        try:
            os.remove(f)
        except OSError:
            pass
    try:
        os.rmdir(tmp)
    except OSError:
        pass


# ---------------- GUI ----------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("MP4 音訊工具 — 去背景音 / 聲場")
        root.geometry("620x730")
        self.q = queue.Queue()
        self.busy = False
        self.player = None
        self.preview_file = os.path.join(tempfile.gettempdir(), "mp4audio_preview.wav")

        pad = dict(padx=10, pady=4)
        frm = ttk.Frame(root)
        frm.pack(fill="both", expand=True)

        # 檔案
        g = ttk.LabelFrame(frm, text="檔案（影片 MP4 或聲音 MP3）")
        g.pack(fill="x", **pad)
        self.path = tk.StringVar()
        ttk.Entry(g, textvariable=self.path).pack(side="left", fill="x", expand=True, padx=6, pady=6)
        ttk.Button(g, text="選擇檔案…", command=self.pick).pack(side="left", padx=6)

        # 分離
        g = ttk.LabelFrame(frm, text="背景音處理")
        g.pack(fill="x", **pad)
        self.sep = tk.StringVar(value="none")
        for label, val in SEP_MODES:
            ttk.Radiobutton(g, text=label, value=val, variable=self.sep).pack(anchor="w", padx=8)
        row = ttk.Frame(g)
        row.pack(fill="x", padx=8, pady=4)
        ttk.Label(row, text="被去掉的部分保留：").pack(side="left")
        self.keep = tk.IntVar(value=0)
        ttk.Scale(row, from_=0, to=50, variable=self.keep, orient="horizontal",
                  command=lambda v: self.keep_lbl.config(text=f"{int(float(v))}%")).pack(side="left", fill="x", expand=True)
        self.keep_lbl = ttk.Label(row, text="0%", width=5)
        self.keep_lbl.pack(side="left")
        self.denoise = tk.BooleanVar(value=False)
        ttk.Checkbutton(g, text="額外降噪（嘶聲、底噪）", variable=self.denoise).pack(anchor="w", padx=8, pady=(0, 6))

        # 聲場
        g = ttk.LabelFrame(frm, text="聲場")
        g.pack(fill="x", **pad)
        row = ttk.Frame(g)
        row.pack(fill="x", padx=8, pady=4)
        ttk.Label(row, text="預設：").pack(side="left")
        self.preset = tk.StringVar(value="原始")
        cb = ttk.Combobox(row, textvariable=self.preset, values=list(PRESETS), state="readonly", width=14)
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda e: self.apply_preset())

        self.vars = {}
        specs = [("width", "立體聲寬度 %", 0, 300), ("surround", "環繞感", 0, 100),
                 ("reverb", "空間殘響", 0, 100), ("balance", "左右平衡", -100, 100),
                 ("bass", "低音 dB", -12, 12), ("treble", "高音 dB", -12, 12),
                 ("voice", "人聲加強 dB", -6, 10), ("volume", "音量 dB", -12, 24)]
        for key, label, lo, hi in specs:
            row = ttk.Frame(g)
            row.pack(fill="x", padx=8, pady=1)
            ttk.Label(row, text=label, width=13).pack(side="left")
            v = tk.IntVar(value=PRESETS["原始"][key])
            lbl = ttk.Label(row, text=str(v.get()), width=5)
            ttk.Scale(row, from_=lo, to=hi, variable=v, orient="horizontal",
                      command=lambda x, v=v, lbl=lbl: (v.set(round(float(x))), lbl.config(text=str(v.get())))
                      ).pack(side="left", fill="x", expand=True)
            lbl.pack(side="left")
            self.vars[key] = (v, lbl)
        self.loudnorm = tk.BooleanVar(value=False)
        ttk.Checkbutton(g, text="自動響度標準化（整支調到串流平台音量，小聲段落拉大）",
                        variable=self.loudnorm).pack(anchor="w", padx=8, pady=(2, 6))

        # 試聽 / 輸出
        g = ttk.Frame(frm)
        g.pack(fill="x", **pad)
        ttk.Label(g, text="試聽起點(秒)：").pack(side="left")
        self.pstart = tk.StringVar(value="30")
        ttk.Entry(g, textvariable=self.pstart, width=6).pack(side="left")
        self.btn_prev = ttk.Button(g, text="▶ 試聽 15 秒", command=self.preview)
        self.btn_prev.pack(side="left", padx=6)
        ttk.Button(g, text="■ 停止", command=self.stop_play).pack(side="left")

        g = ttk.Frame(frm)
        g.pack(fill="x", **pad)
        self.btn_mp4 = ttk.Button(g, text="輸出 MP4", command=lambda: self.export(None))
        self.btn_mp4.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.btn_mp3 = ttk.Button(g, text="只輸出聲音 MP3", command=lambda: self.export(".mp3"))
        self.btn_mp3.pack(side="left", fill="x", expand=True, padx=4)
        self.btn_wav = ttk.Button(g, text="只輸出聲音 WAV", command=lambda: self.export(".wav"))
        self.btn_wav.pack(side="left", fill="x", expand=True, padx=(4, 0))

        self.pb = ttk.Progressbar(frm, maximum=1000)
        self.pb.pack(fill="x", **pad)
        self.status = ttk.Label(frm, text="選一個 MP4 或 MP3 開始。", foreground="#555")
        self.status.pack(anchor="w", **pad)

        if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
            self.path.set(sys.argv[1])
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(100, self.poll)

    # --- helpers
    def params(self):
        p = {k: v.get() for k, (v, _) in self.vars.items()}
        p["loudnorm"] = self.loudnorm.get()
        return p

    def apply_preset(self):
        for k, val in PRESETS[self.preset.get()].items():
            v, lbl = self.vars[k]
            v.set(val)
            lbl.config(text=str(val))

    def pick(self):
        p = filedialog.askopenfilename(filetypes=[
            ("影片或聲音", "*.mp4 *.mov *.mkv *.m4v *.avi *.webm *.mp3 *.wav *.m4a *.flac *.aac *.ogg *.wma"),
            ("影片", "*.mp4 *.mov *.mkv *.m4v *.avi *.webm"),
            ("聲音", "*.mp3 *.wav *.m4a *.flac *.aac *.ogg *.wma"),
            ("全部", "*.*")])
        if p:
            self.path.set(p)

    def set_busy(self, b):
        self.busy = b
        st = "disabled" if b else "normal"
        for btn in (self.btn_prev, self.btn_mp4, self.btn_mp3, self.btn_wav):
            btn.config(state=st)

    def poll(self):
        try:
            while True:
                kind, val = self.q.get_nowait()
                if kind == "log":
                    self.status.config(text=val)
                elif kind == "prog":
                    self.pb["value"] = val * 1000
                elif kind == "done":
                    self.set_busy(False)
                    if val:
                        val()
                elif kind == "err":
                    self.set_busy(False)
                    self.status.config(text="失敗")
                    messagebox.showerror("錯誤", val)
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def check_src(self):
        src = self.path.get().strip().strip('"')
        if not os.path.isfile(src):
            messagebox.showwarning("提示", "請先選擇影片或聲音檔")
            return None
        if not has_audio(src):
            messagebox.showwarning("提示", "這個檔案沒有聲音")
            return None
        return src

    def start_job(self, fn, on_done=None):
        self.set_busy(True)
        self.pb["value"] = 0

        def worker():
            try:
                fn()
                self.q.put(("done", on_done))
            except Exception as e:  # noqa: BLE001
                self.q.put(("err", str(e)))
        threading.Thread(target=worker, daemon=True).start()

    # --- actions
    def preview(self):
        src = self.check_src()
        if not src:
            return
        self.stop_play()
        try:
            st = float(self.pstart.get())
        except ValueError:
            st = 0.0
        dur = probe_duration(src)
        if dur and st >= dur:
            st = max(0.0, dur - 15)
        args = (src, self.preview_file, self.sep.get(), self.keep.get() / 100.0,
                self.params(), self.denoise.get(),
                lambda m: self.q.put(("log", m)), lambda x: self.q.put(("prog", x)))
        self.start_job(lambda: process(*args, start=st, duration=15, audio_only=True),
                       on_done=self.play_preview)

    def play_preview(self):
        self.status.config(text="試聽中…")
        self.player = subprocess.Popen(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", self.preview_file],
            creationflags=CREATE_NO_WINDOW)

    def stop_play(self):
        if self.player and self.player.poll() is None:
            self.player.terminate()
        self.player = None

    def export(self, audio_ext):
        """audio_ext=None 輸出 MP4；'.mp3' / '.wav' 只輸出聲音。"""
        src = self.check_src()
        if not src:
            return
        if audio_ext is None and not has_video(src):
            messagebox.showinfo("提示", "這是聲音檔，沒有畫面可以輸出成 MP4。\n請改按「只輸出聲音 MP3」或「WAV」。")
            return
        audio_only = audio_ext is not None
        base, _ = os.path.splitext(src)
        suffix = {"none": "_聲場", "vocals": "_人聲", "instrumental": "_伴奏", "center_cut": "_去人聲"}[self.sep.get()]
        ext = audio_ext or ".mp4"
        dst = filedialog.asksaveasfilename(
            initialdir=os.path.dirname(src), initialfile=os.path.basename(base) + suffix + ext,
            defaultextension=ext, filetypes=[(ext[1:].upper(), "*" + ext)])
        if not dst:
            return
        if os.path.abspath(dst) == os.path.abspath(src):
            messagebox.showwarning("提示", "不能覆蓋原檔")
            return
        args = (src, dst, self.sep.get(), self.keep.get() / 100.0, self.params(),
                self.denoise.get(), lambda m: self.q.put(("log", m)),
                lambda x: self.q.put(("prog", x)))

        def done():
            self.status.config(text=f"完成：{dst}")
            if messagebox.askyesno("完成", f"已輸出：\n{dst}\n\n要開啟所在資料夾嗎？"):
                subprocess.Popen(["explorer", "/select,", os.path.normpath(dst)])
        self.start_job(lambda: process(*args, audio_only=audio_only), on_done=done)

    def close(self):
        self.stop_play()
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
