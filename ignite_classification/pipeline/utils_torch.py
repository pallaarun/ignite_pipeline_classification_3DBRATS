loss_by_name = {
    "HUBER":"torch.nn.SmoothL1Loss",
    "L2": "torch.nn.MSELoss",
    "L1": "torch.nn.L1Loss",
    "CE": "torch.nn.CrossEntropyLoss"
}

optimizer_by_name = {
    "Adam":"torch.optim.Adam",
}

scheduler_by_name = {
    "ReduceLROnPlateau":"torch.optim.lr_scheduler.ReduceLROnPlateau"
}