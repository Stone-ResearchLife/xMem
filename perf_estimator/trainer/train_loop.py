import torch
from typing import Optional, List
from .plugins import InterfacePlugin


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
        zero_grad_mode: int = 1
):
    """
    Examples:
        zero_grad_mode:
            0: optimizer.zero_grad() before loading dataset
            1: optimizer.zero_grad() just before loss.backward()
            >1: optimizer.zero_grad() not called

    """
    [_plugin.start() for _plugin in plugins]
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
