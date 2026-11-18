# MP4 音訊工具

去人聲／去背景音、聲場調整、音量放大，輸出 MP4 / MP3 / WAV。

## 網頁版

https://sam0819.github.io/mp4-audio-tool/

- 全部在瀏覽器裡用 [ffmpeg.wasm](https://github.com/ffmpegwasm/ffmpeg.wasm) 處理，檔案不會上傳
- 快速去人聲（中置消除）、降噪、聲場預設（寬廣立體聲、虛擬環繞、演唱會廳…）、立體聲寬度、殘響、平衡、低音／高音、人聲加強、音量（最多 +24 dB）、自動響度標準化（-14 LUFS）
- 可輸入 MP4、MOV、MP3、M4A、WAV、FLAC 等；聲音檔只能輸出 MP3 / WAV
- 第一次會下載約 30 MB 的處理引擎，之後由瀏覽器快取
- 檔案太大（超過約 1 GB）時瀏覽器可能記憶體不足

## 桌面版（含 AI 分離）

`mp4audio.py`：tkinter 介面，多了 **AI 保留人聲／AI 去人聲**（[Demucs](https://github.com/facebookresearch/demucs) htdemucs）。

```bash
pip install demucs torchaudio soundfile
python mp4audio.py
```

需要 ffmpeg / ffprobe / ffplay 在 PATH。有 NVIDIA 顯示卡時裝 CUDA 版 PyTorch 會快很多：

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

`vendor/` 內為 @ffmpeg/ffmpeg 0.12.10 與 @ffmpeg/util 0.12.1（MIT）。
