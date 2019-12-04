import time
from options.train_options import TrainOptions
from data import DataLoader
from models import create_model, create_optimizer, create_criterion, save_network
from utils.writer import Writer
from test import run_test
import torch

if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = DataLoader(opt)
    dataset_size = len(dataset)
    print('#training network on = %d images' % dataset_size)

    model = create_model(opt)
    optimizer, scheduler = create_optimizer(model, opt)
    criterion = create_criterion(opt)
    writer = Writer(opt)
    total_steps = 0
    best_acc = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(opt.epochs):
        model.train()
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, (input, label) in enumerate(dataset):
            input,label = input.to(device), label.to(device)
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size

            optimizer.zero_grad()
            pred = model(input)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            if total_steps % opt.print_freq == 0:
                t = (time.time() - iter_start_time) / opt.batch_size
                writer.print_current_losses(epoch, epoch_iter, loss, t, t_data)
                writer.plot_loss(loss, epoch, epoch_iter, dataset_size)

            iter_data_time = time.time()

        if epoch % opt.run_test_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))

            acc = run_test(epoch, model)
            writer.plot_acc(acc, epoch)

            is_best = acc > best_acc
            save_network(model, opt, epoch, is_best)
            model.to(device)

        print('End of epoch %d \t Time Taken: %d sec' %
              (epoch, time.time() - epoch_start_time))
        scheduler.step()

    writer.close()
