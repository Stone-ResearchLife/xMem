import torch
import time
from typing import Optional, List
from transformers import Adafactor
from .plugins import InterfacePlugin


def conv_train_loop(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device,
    optimizer: type(torch.optim.Optimizer),
    iterations: Optional[int] = None,
    lr: float = 0.001,
    plugins: List[InterfacePlugin] = [],
    zero_grad_mode: Optional[int] = None,
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
    zero_grad_mode = 0 if zero_grad_mode > 2 else zero_grad_mode
    [_plugin.start() for _plugin in plugins]
    model.to(device)
    model.train()
    if optimizer is None:
        optimizer = torch.optim.SGD
    optimizer = optimizer(params=model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
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
                if zero_grad_mode == 2:
                    optimizer.zero_grad()
                scheduler.step()
                print(
                    f"Epoch: {epoch}/{index}, learning rate: {scheduler.get_last_lr()}"
                )

                if iterations is not None and index >= iterations:
                    break
    except Exception as e:
        raise RuntimeError(f"Training failed: {e}") from e

    finally:
        for _plugin in plugins:
            print(f"Stop {_plugin.tool_name} plugin")
            _plugin.stop()


def transformer_train_loop(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device,
    optimizer: type(torch.optim.Optimizer),
    iterations: Optional[int] = 2,
    lr: float = 0.001,
    plugins: List[InterfacePlugin] = [],
    zero_grad_mode: Optional[int] = None,
):

    [_plugin.start() for _plugin in plugins]
    time.sleep(2)
    zero_grad_mode = zero_grad_mode or 0
    zero_grad_mode = 0 if zero_grad_mode > 2 else zero_grad_mode
    if optimizer.__name__ == "Adafactor":
        optimizer = optimizer(params=model.parameters(), lr=lr, relative_step=False)
    else:
        optimizer = optimizer(params=model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    try:
        model.to(device)
        for epoch in range(epochs):
            for index, batch in enumerate(data_loader):
                [_plugin.step() for _plugin in plugins]
                if zero_grad_mode == 0:
                    optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    if zero_grad_mode == 1:
                        optimizer.zero_grad()
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                if zero_grad_mode == 2:
                    optimizer.zero_grad()
                if index == iterations:
                    break
                scheduler.step()
    except Exception as e:
        raise RuntimeError(f"Training failed: {e}") from e
    finally:
        [_plugin.stop() for _plugin in plugins]


def transformer_mixed_precision_train_loop(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device,
    optimizer: type(torch.optim.Optimizer),
    iterations: Optional[int] = 2,
    lr: float = 0.001,
    plugins: List[InterfacePlugin] = [],
    zero_grad_mode: Optional[int] = None,
):

    [_plugin.start() for _plugin in plugins]
    time.sleep(2)
    zero_grad_mode = zero_grad_mode or 0
    zero_grad_mode = 0 if zero_grad_mode > 2 else zero_grad_mode
    if isinstance(optimizer, type(Adafactor)):
        optimizer = optimizer(params=model.parameters(), lr=lr, relative_step=False)
    else:
        optimizer = optimizer(params=model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    try:
        model.to(device)
        for epoch in range(epochs):
            for index, batch in enumerate(data_loader):
                [_plugin.step() for _plugin in plugins]
                if zero_grad_mode == 0:
                    optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with torch.amp.autocast("cpu", dtype=torch.float16):
                        outputs = model(**batch)
                        if zero_grad_mode == 1:
                            optimizer.zero_grad()
                        loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                if zero_grad_mode == 2:
                    optimizer.zero_grad()
                if index == iterations:
                    break
                scheduler.step()
    except Exception as e:
        raise RuntimeError(f"Training failed: {e}") from e
    finally:
        [_plugin.stop() for _plugin in plugins]

