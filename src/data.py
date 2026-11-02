"""Завантаження Google Speech Commands і Mel-спектрограми (спільні для train і API)."""
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

SAMPLE_RATE = 16000
N_SAMPLES = 16000  # 1 секунда

_mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_fft=400, hop_length=160, n_mels=40
)
_to_db = torchaudio.transforms.AmplitudeToDB()


def to_mono_16k(wave: np.ndarray, sr: int) -> torch.Tensor:
    """numpy-сигнал (n,) або (n, ch) -> моно, 16 кГц, рівно 1 с."""
    x = torch.from_numpy(np.asarray(wave, dtype=np.float32))
    if x.ndim == 2:
        x = x.mean(dim=1)
    if sr != SAMPLE_RATE:
        x = torchaudio.functional.resample(x, sr, SAMPLE_RATE)
    if x.numel() < N_SAMPLES:
        x = F.pad(x, (0, N_SAMPLES - x.numel()))
    else:
        x = x[:N_SAMPLES]
    return x


def waveform_to_features(x: torch.Tensor) -> torch.Tensor:
    """(16000,) -> лог-Mel спектрограма [1, 40, 101], нормалізована."""
    f = _to_db(_mel(x))
    f = (f - f.mean()) / (f.std() + 1e-5)
    return f.unsqueeze(0)


class KWSDataset(torch.utils.data.Dataset):
    """Підмножина Speech Commands для обраних класів; ознаки кешуються в RAM."""

    def __init__(self, root: str, split: str, classes: list[str]):
        subset = {"train": "training", "val": "validation", "test": "testing"}[split]
        Path(root).mkdir(parents=True, exist_ok=True)
        ds = torchaudio.datasets.SPEECHCOMMANDS(root, download=True, subset=subset)
        self.classes = classes
        self.paths = [p for p in ds._walker if Path(p).parent.name in classes]
        self.features, self.targets = [], []
        for p in self.paths:
            wave, sr = sf.read(p, dtype="float32")
            self.features.append(waveform_to_features(to_mono_16k(wave, sr)))
            self.targets.append(classes.index(Path(p).parent.name))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.features[i], self.targets[i]
