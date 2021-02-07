import torch


def train_one_epoch(model, 
                    optimizer, 
                    data_loader, 
                    criterion_fn, 
                    epoch, 
                    print_every=20,
                    device=torch.device('cpu')):

    running_loss = 0.
    print(f'--- Training Epoch {epoch} ---')

    model.train()
    model.to(device)
    model.zero_grad()

    for i, (x, y) in enumerate(data_loader):

        x = x.to(device)
        y = y.to(device)

        prediction = model(x, y)
        print(prediction.size(), y.size())
        
        loss = criterion_fn(prediction, y)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % print_every == 0:
            loss_mean = running_loss / i
            print(f'Epoch [{epoch}] loss: {loss_mean:.4f}')


@torch.no_grad()
def evaluate(model,
             data_loader,
             criterion_fn,
             metrics=[],
             decode_fn=None,
             device=torch.device('cpu')):

    print(f'--- Evaluate ---')

    model.eval()
    model.to(device)

    running_loss = 0.
    running_metrics = {m.__name__: 0. for m in metrics}

    for x, y in data_loader:

        x = x.to(device)
        y = y.to(device)

        prediction = model(x)
        prediction_ids = (prediction.argmax(-1) 
                          if decode_fn is None else 
                          decode_fn(prediction))

        loss = criterion_fn(prediction, y)
        current_metrics = {m.__name__: m(prediction_ids, y) for m in metrics}

        running_loss += loss.item()
        running_metrics += {m.__name__: running_metrics[m.__name__] + v 
                            for m, v in current_metrics.items()}

    n = len(data_loader)
    loss_mean = running_loss / n
    avg_metrics = {m.__name__: v / n for m, v in running_metrics.items()}
    avg_metrics = ' '.join(f'{m.__name__}: {v:.4f}' 
                            for m, v in avg_metrics.items())

    print(f'Evaluation loss: {loss_mean:.4f} {avg_metrics}')
