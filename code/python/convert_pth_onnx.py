import torch
import torch.nn as nn
from pathlib import Path


class SEBlock(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c // r, 1),
            nn.SiLU(),
            nn.Conv2d(c // r, c, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(x)
        return x * w


def conv_bn(in_c, out_c, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, s, p, bias=False),
        nn.BatchNorm2d(out_c),
        nn.SiLU(inplace=True)
    )


class ResBlock(nn.Module):
    def __init__(self, c, down=False):
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

    def forward(self, x):
        idt = self.down(x) if isinstance(self.down, nn.AvgPool2d) else x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        if out.shape == idt.shape:
            out = out + idt
        return self.act(out)


class ResMelNet(nn.Module):
    def __init__(self, n_classes: int):
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
            nn.Conv2d(64, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True)
        )
        self.proj3 = nn.Sequential(
            nn.Conv2d(128, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.proj2(x)
        x = self.stage3(x)
        x = self.proj3(x)
        x = self.stage4(x)
        return self.head(x)


ckpt_path = Path("data/checkpoints/guitar_macro_classifier.pth")

ckpt = torch.load(ckpt_path, map_location="cpu")
state = ckpt["model"]

if any(k.startswith("_orig_mod.") for k in state.keys()):
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

classes = ckpt.get("classes", [])
n_classes = len(classes) if classes else state["head.6.weight"].shape[0]

model = ResMelNet(n_classes=n_classes)
model.load_state_dict(state, strict=True)
model.eval()
model.to("cpu")

print(f"Loaded model with {n_classes} classes from {ckpt_path}")

dummy_input = torch.randn(1, 1, 256, 512, dtype=torch.float32)

onnx_path = ckpt_path.with_suffix(".onnx")

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    do_constant_folding=True,
    input_names=["mel"],
    output_names=["logits"],
    dynamic_axes={
        "mel":    {0: "batch", 2: "freq", 3: "time"},
        "logits": {0: "batch"}
    },
    opset_version=17,
)

print("Saved ONNX model to:", onnx_path)