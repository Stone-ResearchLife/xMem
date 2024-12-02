import torch
from typing import Optional, List
from .plugins import InterfacePlugin
from ..dataset import audio_dataset


def conv_train_loop(
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        epochs: int,
        device: torch.device,
        batch_size: int = None,
        iterations: Optional[int] = None,
        loss: Optional[torch.nn.Module] = None,
        lr: float = 0.001,
        plugins: List[InterfacePlugin] = [],
        optimizer: torch.optim.Optimizer = None,
        zero_grad_mode: Optional[int] = None
):
    """
    Examples:
        zero_grad_mode:
            0: optimizer.zero_grad() before loading dataset
            1: optimizer.zero_grad() after loading dataset
            2: optimizer.zero_grad() just before loss.backward()
            3: optimizer.zero_grad() in end of each iteration

    """
    zero_grad_mode = zero_grad_mode or 0
    [_plugin.start() for _plugin in plugins]
    # [_plugin.step() for _plugin in plugins]
    model.to(device)
    model.train()
    if optimizer is None:
        optimizer = torch.optim.SGD
    optimizer = optimizer(params=model.parameters(), lr=lr)
    criterion = loss or torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    try:
        for epoch in range(epochs):
            for index, (inputs, labels) in enumerate(data_loader):
                [_plugin.step() for _plugin in plugins]
                if zero_grad_mode == 0:
                    optimizer.zero_grad()
                inputs, labels = inputs.to(device), labels.to(device)

                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    loss_value = criterion(outputs, labels)
                    if zero_grad_mode == 1:
                        optimizer.zero_grad()
                    loss_value.backward()
                    optimizer.step()
                _, preds = torch.max(outputs, 1)
                scheduler.step()
                print(f"Epoch: {epoch}/{index}, learning rate: {scheduler.get_last_lr()}")

                if iterations is not None and index >= iterations:
                    break
    finally:
        for _plugin in plugins:
            print(f"Stop {_plugin.tool_name} plugin")
            _plugin.stop()


def audio_train_loop(
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        epochs: int,
        device: torch.device,
        batch_size: int = None,
        iterations: Optional[int] = None,
        loss: Optional[torch.nn.Module] = None,
        lr: float = 0.001,
        plugins: List[InterfacePlugin] = [],
        optimizer: torch.optim.Optimizer = None,
        zero_grad_mode: int = 1

):
    if data_loader is None:
        data_loader, vocab_size = audio_dataset(batch_size=batch_size, sample_rate=16000, n_mels=128, float16=False)
    [_plugin.start() for _plugin in plugins]
    model.to(device)
    model.train()
    if optimizer is None:
        optimizer = torch.optim.SGD
    optimizer = optimizer(params=model.parameters(), lr=lr)
    criterion = loss or torch.nn.CTCLoss(blank=0, zero_infinity=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    try:
        for epoch in range(epochs):
            total_loss = 0
            for index, (inputs, input_lengths, targets, target_lengths) in enumerate(data_loader):
                [_plugin.step() for _plugin in plugins]
                if zero_grad_mode == 0:
                    optimizer.zero_grad()
                inputs = inputs.to(device)
                input_lengths = input_lengths.to(device)
                targets = targets.to(device)
                target_lengths = target_lengths.to(device)
                if zero_grad_mode == 1:
                    optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    outputs, output_lengths = model(inputs, input_lengths)  # outputs: (batch size, time, input_dim)
                    outputs = outputs.permute(1, 0, 2)  # (time, batch_size, input_dim)
                    loss = criterion(outputs.log_softmax(2), targets, output_lengths, target_lengths)
                    if zero_grad_mode == 2:
                        optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                scheduler.step()

                if zero_grad_mode == 3:
                    optimizer.zero_grad()

                if iterations is not None and index >= iterations:
                    break
    finally:
        for _plugin in plugins:
            print(f"Stop {_plugin.tool_name} plugin")
            _plugin.stop()

