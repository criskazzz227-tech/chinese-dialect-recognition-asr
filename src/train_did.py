import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import APP, DIALECTS, DID_CKPT, TEST_CSV, TRAIN_CSV
from .logging_config import configure_logging
from .features import cmvn, logmel
from .io_audio import load_wav, pad_or_trim, simple_vad_trim
from .model_did import DID_CNN

label2id = {k: i for i, k in enumerate(DIALECTS)}


class DialectDataset(Dataset):
    def __init__(self, csv_path: str, augment: bool = False):
        df = pd.read_csv(csv_path)
        self.paths = df["path"].tolist()
        self.labels = [label2id[x] for x in df["label"].tolist()]
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        audio = load_wav(self.paths[idx])
        audio = simple_vad_trim(audio)
        if self.augment:
            audio = augment_audio(audio)
        audio = pad_or_trim(audio)
        feat = cmvn(logmel(audio))
        label = self.labels[idx]
        return torch.tensor(feat), torch.tensor(label)


def augment_audio(audio):
    if len(audio) == 0:
        return audio

    gain = np.random.uniform(0.75, 1.25)
    audio = audio * gain

    if np.random.random() < 0.5:
        noise_std = np.random.uniform(0.001, 0.008)
        audio = audio + np.random.normal(0.0, noise_std, size=audio.shape).astype(np.float32)

    if np.random.random() < 0.5 and len(audio) > 16000:
        max_shift = min(len(audio) // 10, 8000)
        shift = np.random.randint(-max_shift, max_shift + 1)
        audio = np.roll(audio, shift)

    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def run_train(epochs=30, batch_size=16, lr=5e-4, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    tr = DialectDataset(TRAIN_CSV, augment=True)
    te = DialectDataset(TEST_CSV, augment=False)
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=0)
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=0)

    model = DID_CNN(in_ch=80, num_classes=len(DIALECTS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    best_acc = 0.0
    for ep in range(1, epochs + 1):
        model.train()
        loss_sum, n = 0.0, 0
        for feat, y in tqdm(tr_loader, desc=f"Train ep{ep}"):
            feat, y = feat.to(device), y.to(device)
            opt.zero_grad()
            logits = model(feat)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(y)
            n += len(y)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for feat, y in te_loader:
                feat, y = feat.to(device), y.to(device)
                pred = model(feat).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += len(y)

        acc = correct / max(total, 1)
        print(f"[ep{ep}] train_loss={loss_sum / max(n, 1):.4f} test_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "metadata": {
                        "model_type": "logmel_1d_cnn",
                        "version": f"cnn-{APP.version}",
                        "labels": DIALECTS,
                        "best_accuracy": acc,
                        "created_at": datetime.now(
                            timezone.utc
                        ).isoformat(),
                    },
                },
                DID_CKPT,
            )
            print("Saved:", DID_CKPT)

    print("Best checkpoint:", DID_CKPT)
    print("Best test acc:", best_acc)


if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()
    run_train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
