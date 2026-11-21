"""Training loops shared by associative-memory experiments."""


def train_epoch(model, optimizer, criterion, keys, values, order):
    """Update once per index, in the supplied order (including repeats).

    Inputs are tensors on the model device. The caller owns the optimizer,
    random order and number of epochs; optimizer state survives across calls.
    """
    if len(keys) != len(values):
        raise ValueError("keys and values must have the same length")
    model.train()
    for i in order:
        optimizer.zero_grad()
        loss = criterion(model(q=keys[i]), values[i])
        loss.backward()
        optimizer.step()
