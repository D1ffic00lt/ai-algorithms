import torch

from typing import Any
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from sklearn.metrics import accuracy_score

try:
    from IPython.display import clear_output
except ImportError | ModuleNotFoundError:
    from IPython.core.display_functions import clear_output


class BaseModel(torch.nn.Module):
    DEVICE = "cuda"

    def __init__(self, device: str = "cuda", *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.DEVICE = device

        if not torch.cuda.is_available():
            self.DEVICE = "cpu"

    @torch.inference_mode()
    def evaluate(self, dataloader):
        self.eval()
        num_val_batches = len(dataloader)

        real = []
        preds = []

        with torch.autocast(self.DEVICE):
            for X, y in tqdm(
                    dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False
            ):
                image = X[0].to(device=self.DEVICE, dtype=torch.float32)
                desc = X[1].to(device=self.DEVICE, dtype=torch.float32)
                y = y.to(device=self.DEVICE, dtype=torch.long)

                pred = self((image, desc))

                real.append(y)
                preds.append(torch.max(pred, 1))
                print(real, preds)

        return real, preds

    @torch.inference_mode()
    def predict_one_sample(self, item):
        self.eval()

        with torch.autocast(self.DEVICE):
            pred = self(item)

        return torch.max(pred, 1)

    def __fit_epoch(self, train_loader, criterion, optimizer, grad_scaler=None):
        self.train()

        running_loss = 0.0
        processed_data = 0

        for inputs, labels in tqdm(train_loader):
            image, desc = inputs
            image = image.to(self.DEVICE)
            desc = desc.to(self.DEVICE)
            labels = labels.to(self.DEVICE)
            optimizer.zero_grad()

            outputs = self((image, desc))
            # outputs = F.sigmoid(outputs)

            loss = criterion(outputs.to(self.DEVICE), labels)

            if grad_scaler is not None:
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss.backward()

            optimizer.step()

            running_loss += loss.item()
            processed_data += 1

        train_loss = running_loss / processed_data
        train_acc = self.evaluate(train_loader)
        return train_loss, train_acc

    def __eval_epoch(self, val_loader, criterion, scheduler=None):
        self.eval()
        running_loss = 0.0
        processed_size = 0

        for inputs, labels in tqdm(val_loader):
            inputs = inputs.to(self.DEVICE)
            labels = labels.to(self.DEVICE)

            with torch.set_grad_enabled(False):
                outputs = self(inputs)

                loss = criterion(outputs.to(self.DEVICE), labels.to(self.DEVICE))

            running_loss += loss.item()
            processed_size += 1
        val_loss = running_loss / processed_size

        val_acc = self.evaluate(val_loader)

        if scheduler is not None:
            scheduler.step(val_acc)

        return val_loss, val_acc

    def fit(self, train_data, val_data, epochs, batch_size,
            scheduler: Any = None, optimizer: Any = None, grad_scaler: Any = None,
            criterion: Any = nn.NLLLoss()):
        train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_data = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        history = []
        log_template = "\nEpoch {ep:03d} train_loss: {t_loss:0.4f} \
        val_loss {v_loss:0.4f} train_acc {t_acc:0.4f} val_acc {v_acc:0.4f}"
        with tqdm(desc="epoch", total=epochs) as pbar_outer:
            if not optimizer:
                optimizer = torch.optim.Adam(self.parameters(), amsgrad=True, lr=0.0005)
            try:
                for epoch in range(epochs):
                    if epoch % 10 == 0 and epoch != 0:
                        torch.save(
                            self.state_dict(), 'modelseg.pth'
                        )
                    train_loss, train_acc = self.__fit_epoch(
                        train_data, criterion, optimizer, grad_scaler=grad_scaler
                    )
                    clear_output(wait=True)
                    tqdm.write(f"loss: {train_loss}")

                    tqdm.write(str(self.metric))

                    val_loss, val_acc = self.__eval_epoch(val_data, criterion, scheduler=scheduler)

                    history.append((train_loss, train_acc, val_loss, val_acc))

                    tqdm.write(log_template.format(ep=epoch + 1, t_loss=train_loss,
                                                   v_loss=val_loss, t_acc=train_acc, v_acc=val_acc))
                    pbar_outer.update()

            except KeyboardInterrupt:
                return history
        return history
