"""Track B: deep audio CNN on mel-spectrograms."""
import sys
from pathlib import Path

# Add both ml/ (for config) and ml/src/ (for sibling modules) to sys.path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from feature_extraction import wav_to_melspec
from ml_config import MODELS_DIR, RAW_DIR, RANDOM_STATE


class VoiceDataset(Dataset):
    def __init__(self, files, labels):
        self.files, self.labels = files, labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        x = torch.tensor(wav_to_melspec(self.files[i])).unsqueeze(0)
        return x, torch.tensor(self.labels[i], dtype=torch.float32)


class NeuroEchoCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.4), nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def main():
    pd_files = glob.glob(str(RAW_DIR / "audio" / "pd" / "*.wav"))
    healthy_files = glob.glob(str(RAW_DIR / "audio" / "healthy" / "*.wav"))
    files = pd_files + healthy_files
    labels = [1] * len(pd_files) + [0] * len(healthy_files)

    if not files:
        print(f"No audio files found. Add WAVs to {RAW_DIR / 'audio'}/{{pd,healthy}}/")
        return

    f_tr, f_te, l_tr, l_te = train_test_split(
        files, labels, test_size=0.2, stratify=labels, random_state=RANDOM_STATE)
    train_dl = DataLoader(VoiceDataset(f_tr, l_tr), batch_size=16, shuffle=True)
    test_dl = DataLoader(VoiceDataset(f_te, l_te), batch_size=16)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NeuroEchoCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(20):
        model.train()
        total = 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"Epoch {epoch+1:02d}  loss {total/len(train_dl):.4f}")

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            pred = (torch.sigmoid(model(x)) > 0.5).float()
            correct += (pred == y).sum().item()
            total += len(y)
    print("Test accuracy:", round(correct / total, 3))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / "cnn_model.pt")
    print("Saved model -> models/cnn_model.pt")


if __name__ == "__main__":
    main()
