import torch


def train_one_epoch(model, 
                    optimizer, 
                    data_loader, 
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

        loss, prediction = model(x, y)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % print_every == 0:
            loss_mean = running_loss / (i + 1)
            print(f'Epoch [{epoch}] [{i}/{len(data_loader)}] '
                  f'loss: {loss_mean:.4f}')


@torch.no_grad()
def evaluate(model,
             data_loader,
             metrics=[],
             device=torch.device('cpu')):

    print(f'--- Evaluate ---')

    model.eval()
    model.to(device)

    running_loss = 0.
    running_metrics = {m.__name__: 0. for m in metrics}

    for x, y in data_loader:

        x = x.to(device)
        y = y.to(device)

        loss, prediction = model(x)

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
