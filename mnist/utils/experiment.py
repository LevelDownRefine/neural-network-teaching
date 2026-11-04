"""Common epoch and model-selection flow; model-specific work stays in callbacks."""
from contextlib import nullcontext


def run_epochs(start, epochs, best, history, best_checkpoint, *, train_epoch,
               validate, capture, record, save_epoch, validation_context=nullcontext,
               replace_on_tie=False):
    for epoch in range(start, epochs):
        training = train_epoch(epoch)
        with validation_context():
            validation = validate(epoch)
            score = validation['accuracy']
            if score > best or (replace_on_tie and score == best):
                best = score
                best_checkpoint = capture(epoch)
        history.append(record(epoch, training, validation))
        save_epoch(epoch, best, history, best_checkpoint)
    return best, history, best_checkpoint
