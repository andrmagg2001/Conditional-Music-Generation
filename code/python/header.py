import os
import random
import torch
import json
import torch._dynamo
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import amp
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from contextlib import nullcontext


SEED = 1337
"""The **SEED** ensures reproducible randomness across Python, NumPy, and PyTorch."""

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch._dynamo.config.disable = True
torch._dynamo.config.suppress_errors = True

DEVICE      = "cpu"
"""**DEVICE** is the compute device (*CUDA*, *MPS* (Apple Silicon), *CPU*) where tensors and the model are placed."""

DEVICE_TYPE = "cpu"
"""**DEVICE_TYPE** is the string label indicating which backend is active (*CUDA*, *MPS* (Apple Silicon), *CPU*)."""

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DEVICE_TYPE = "cuda"
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    DEVICE_TYPE = "mps"
else:
    DEVICE = torch.device("cpu")
    DEVICE_TYPE = "cpu"

if DEVICE_TYPE == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

CACHE_ROOT  = "../../data/dataset/guitar_mel"
"""**CACHE_ROOT** is the filesystem directory where all precomputed mel-spectrogram. `.npy` files and `index.json` are stored."""

CHECKPOINTS = "../../data/checkpoints"
"""**CHECKPOINTS** is the directory where trained model `.pth` are saved."""


IDX = ""
"""**IDX** loads the dataset index from a JSON file."""

with open(f"{CACHE_ROOT}/index.json") as f:
    IDX = json.load(f)

CLASSES = tuple(IDX["classes"])
"""**CLASSES** extracts the class names from the index and stores them as an immutable tuple."""

ITEMS   = IDX["items"]
"""**ITEMS** loads the list of dataset samples (each with paths and metadata) from the index file."""

P       = IDX.get("params", {})
"""**P** retrivers optional preprocessing parameters stored in the index file."""

DB_LO   = float(P.get("db_lo", -80.0))
"""**DB_LO** Lower Bound in dB used to normalize mel-spectrogram values."""

DB_HI   = float(P.get("db_hi", 0.0))
"""**DB_HI** Upper Bound in dB used to mel-spectrogram values."""

_sr  = float(P.get("sr", 44100.0))
"""**Sample rate** (Hz) used during mel-spectrogram computation and audio processing."""

_hop = float(P.get("hop", 256.0))
"""**Hop length** (number of audio samples between STFT frames) used when computing mel-spectrograms."""

FPS  = _sr / _hop
"""**FPS** (Frames per second) of mel-spectrogram (how many time-frames are produced each second)"""

EPOCHS  = 50
"""**EPOCHS** is the number of full passes the training loop makes over the entire dataset."""

WD      = 1e-4
"""**Weight decay** coefficient used for L2 regularization during optimization."""

LAB_SM  = 0.10
"""**Label-Smoothing** factor controlling how much the true label is softened in the loss."""

T_SEC   = 6.0
"""**Duration** of each mel-spectrogram crop fed to the model."""

TIME_D  = 16
"""**TIME_D** is the downsampling factor applied along the *time axis* of the mel-spectrogram."""

FREQ_D  = 4
"""**FREQ_D** is the Downsampling factor applied along the *frequency* axis of the mel-spectrogram."""

BATC_S  = 4
"""**BATC_S** is the batch size used for training and validation."""

ACC_ST  = 4
"""**ACC_ST** is the gradient accumulation steps."""

LR      = 3e-4
"""**LR** is the learning rate for the optimizer."""

PATIEN  = 10
"""**PATIENT** is the early-stopping patience."""

NUM_WORKERS = 0
"""**NUM_WORKERS** is the number of subprocesses used by the DataLoader."""

PIN_MEMORY  = False
"""**PIN_MEMORY** wheter DataLoader should use pinned memory for faster GPU transfers."""


if DEVICE_TYPE == "cuda":
    T_SEC   = 20.0
    TIME_D  = 12
    FREQ_D  = 4
    BATC_S  = 6
    ACC_ST  = 2
    LR      = 5e-4
    PATIEN  = 12
    NUM_WORKERS = 6
    PIN_MEMORY  = True

elif DEVICE_TYPE == "mps":
    T_SEC   = 6.0
    TIME_D  = 12
    FREQ_D  = 4
    BATC_S  = 6
    ACC_ST  = 2
    LR      = 5e-4
    PATIEN  = 12
    NUM_WORKERS = 0
    PIN_MEMORY  = False



CROP_F = int(FPS * T_SEC)
"""**CROP_F** is the number of time frames to extract from each mel-spectrogram crop based on the target duration."""

def _norm_db(x: float) -> float:
    """
    Normalize a dB-scaled spectrogram value to the [0,1] range.

    Parameters:
        x (float or np.array): Input value(s) in db scale.

    Returns:
        float or np.ndarray: Normalized value(s) in the [0,1] range.
    """
    return (x - DB_LO) / (DB_HI - DB_LO + 1e-8)


class MelDataset(Dataset):
    """
    Dataset for cached log-mel spectrogram patches and macro-genre labels.
    """

    def __init__(self, items: list[dict], train: bool = True) -> None:
        """
        Initialize the MelDataset.

        Parameters:
            items (list[dict]): List of metadata entries, each with at least
                'mel_path' (str) and 'class_idx' (int).
            train (bool): If True, enables random cropping and augmentation;
                if False, uses a centered deterministic crop.
        """
        self.items = items
        self.train = train
    
    def __len__(self) -> int:
        """
        Return the total number of items in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.items)

    def _rand_crop_cols(self, T: int) -> tuple[int, int]:
        """
        Sample a random time-window (in frames) for cropping.

        Parameters:
            T (int): Total number of time frames in the spectrogram.

        Returns:
            tuple[int, int]: Start and end frame indices [s, e) for the crop.
        """
        if T <= CROP_F:
            return 0, T
        s = random.randint(0, T - CROP_F)
        return s, s + CROP_F
        
    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        """
        Load one mel-spectrogram patch and its label.

        Parameters:
            i (int): Index of the sample to retrieve.

        Returns:
            tuple[torch.Tensor, int]:
                - X: Tensor of shape (1, F, T_crop), normalized to [0, 1],
                  with optional time/frequency masking and downsampling.
                - y: Integer class index corresponding to the macro-genre.
        """
        it = self.items[i]
        X = np.load(it["mel_path"]).astype(np.float32)
        X = _norm_db(X)
        M, T = X.shape

        if self.train:
            s, e = self._rand_crop_cols(T)
        else:
            s = max(0, T // 2 - CROP_F // 2)
            e = min(T, T // 2 + CROP_F // 2)
        X = X[:, s:e]

        if X.shape[1] < CROP_F:
            pad = CROP_F - X.shape[1]
            X = np.pad(X, ((0, 0), (0, pad)), mode="edge")

        X = torch.from_numpy(X).unsqueeze(0)

        if self.train:
            t_w = int(0.08 * X.shape[-1])
            if t_w > 0:
                t0 = random.randint(0, max(0, X.shape[-1] - t_w))
                X[:, :, t0:t0 + t_w] = 0.0

            f_w = int(0.06 * X.shape[-2])
            if f_w > 0:
                f0 = random.randint(0, max(0, X.shape[-2] - f_w))
                X[:, f0:f0 + f_w, :] = 0.0

        if FREQ_D > 1 or TIME_D > 1:
            X = F.avg_pool2d(X, kernel_size=(FREQ_D, TIME_D), stride=(FREQ_D, TIME_D))

        y = it["class_idx"]
        return X, y
    

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) block for channel-wise attention on 2D feature maps.
    """

    def __init__(self, c: int, r: int = 16) -> None:
        """
        Initialize the SEBlock.

        Parameters:
            c (int): Number of input and output channels.
            r (int): Reduction ratio for the inner bottleneck (c -> c // r -> c).
        """
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c // r, 1),
            nn.SiLU(),
            nn.Conv2d(c // r, c, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel-wise attention to the input feature map.

        Parameters:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Reweighted tensor of shape (B, C, H, W),
            where each channel is scaled by a learned attention weight.
        """
        w = self.fc(x)
        return x * w


def conv_bn(
    in_c: int,
    out_c: int,
    k: int = 3,
    s: int = 1,
    p: int = 1
) -> nn.Sequential:
    """
    Build a Conv2d → BatchNorm2d → SiLU block.

    Parameters:
        in_c (int): Number of input channels.
        out_c (int): Number of output channels.
        k (int): Convolution kernel size.
        s (int): Convolution stride.
        p (int): Convolution padding.

    Returns:
        nn.Sequential: A sequential module with:
            Conv2d(in_c → out_c) → BatchNorm2d(out_c) → SiLU activation.
    """
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, s, p, bias=False),
        nn.BatchNorm2d(out_c),
        nn.SiLU(inplace=True),
    )


class ResBlock(nn.Module):
    """
    Residual convolutional block with optional downsampling and Squeeze-and-Excitation.

    Parameters:
        c (int): Number of input and output channels.
        down (bool): If True, applies spatial downsampling by a factor of 2
            via stride in conv1 and an AvgPool2d on the residual path.
    """
    def __init__(self, c: int, down: bool = False) -> None:
        super().__init__()
        s = 2 if down else 1
        self.conv1 = conv_bn(c, c, 3, s, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c)
        )
        self.se = SEBlock(c)
        self.down = nn.AvgPool2d(2) if down else nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the residual block.

        Parameters:
            x (torch.Tensor): Input feature map of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output feature map of shape (B, C, H_out, W_out),
            potentially downsampled if `down=True`.
        """
        idt = self.down(x) if isinstance(self.down, nn.AvgPool2d) else x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        if out.shape == idt.shape:
            out = out + idt
        return self.act(out)


class ResMelNet(nn.Module):
    """
    CNN classifier for log-mel spectrograms with residual and SE blocks.

    Parameters:
        n_classes (int): Number of output classes for the final classifier head.
    """
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            conv_bn(1, 64, 5, 2, 2),
            nn.MaxPool2d(2)
        )
        self.stage1 = nn.Sequential(*[ResBlock(64) for _ in range(3)])
        self.stage2 = nn.Sequential(
            ResBlock(64, down=True),
            *[ResBlock(64) for _ in range(3)]
        )
        self.stage3 = nn.Sequential(
            ResBlock(128, down=True),
            *[ResBlock(128) for _ in range(3)]
        )
        self.stage4 = nn.Sequential(
            ResBlock(256, down=True),
            *[ResBlock(256) for _ in range(3)]
        )
        self.proj2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True)
        )
        self.proj3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of ResMelNet.

        Parameters:
            x (torch.Tensor): Input batch of spectrograms with shape (B, 1, M, T),
                where M is the mel dimension and T is time.

        Returns:
            torch.Tensor: Logits of shape (B, n_classes) for classification.
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.proj2(x)
        x = self.stage3(x)
        x = self.proj3(x)
        x = self.stage4(x)
        return self.head(x)
    

class LSCELoss(nn.Module):
    """
    Label-smoothed cross entropy loss.

    Parameters:
        eps (float): Smoothing factor in [0, 1]. 0 = no smoothing,
            higher values distribute more probability mass to non-target classes.
    """
    def __init__(self, eps: float = 0.1) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute label-smoothed cross entropy loss.

        Parameters:
            logits (torch.Tensor): Model outputs of shape (B, C),
                where B is batch size and C is number of classes.
            y (torch.Tensor): Ground-truth class indices of shape (B,).

        Returns:
            torch.Tensor: Scalar tensor containing the mean loss over the batch.
        """
        num_classes = logits.size(-1)
        y_onehot = torch.zeros_like(logits).scatter_(1, y.unsqueeze(1), 1)
        y_smooth = (1 - self.eps) * y_onehot + self.eps / num_classes
        logp = F.log_softmax(logits, dim=1)
        return -(y_smooth * logp).sum(dim=1).mean()
