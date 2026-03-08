import httpx
import torch
from models import MLP
from torch import nn
from torch.utils.data import DataLoader, Dataset
from utils import create_dataset, transform_to_tensor

model = MLP(in_features=145, hidden_features=70, out_features=1)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


class BookDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X, y = self.data[index]
        return torch.from_numpy(X).float(), torch.from_numpy(y).reshape(1).float()


def create_dataloader(text):
    dataset = [
        (transform_to_tensor(sequence), label)
        for sequence, label in create_dataset(text)
    ]
    split = int(0.8 * len(dataset))
    train_ds, val_ds = dataset[:split], dataset[split:]
    train_loader = DataLoader(BookDataset(train_ds), shuffle=True, batch_size=256)
    val_loader = DataLoader(BookDataset(val_ds), shuffle=True, batch_size=256)
    return train_loader, val_loader


if __name__ == "__main__":
    url = "https://www.gutenberg.org/files/2701/2701-0.txt"  # Moby-Dick
    text = httpx.get(url, timeout=30).text
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device=device)

    train_loader, val_loader = create_dataloader(text)
    epochs = 5

    for i in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device=device, dtype=torch.float32)
            y_batch = y_batch.to(device=device, dtype=torch.float32)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device=device, dtype=torch.float32)
                y_batch = y_batch.to(device=device, dtype=torch.float32)

                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss

        print(
            f"Epoch {i} | Train Loss: {train_loss / len(train_loader):.4f} | Val Loss: {val_loss / len(val_loader):.4f}"
        )
