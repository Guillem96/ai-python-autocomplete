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
    model.zero_grad()

    for i, (x, y) in enumerate(data_loader):

        x = x.to(device)
        y = y.to(device)

        prediction = model(x)

        loss = criterion_fn(prediction, y)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % print_every == 0:
            loss_mean = running_loss / i
            print(f'Epoch [{epoch}] loss: {loss_mean:.4f}')
