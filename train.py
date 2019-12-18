# import comet_ml in the top of your file
from comet_ml import Experiment

import time
from options.train_options import TrainOptions
from data import DataLoader
from models import create_model, create_optimizer, save_network
from loss import create_criterion
from utils.writer import Writer
from test import run_test
import torch
from utils.evaluate import evaluate

if __name__ == '__main__':

    # Add the following code anywhere in your machine learning file
    experiment = Experiment(api_key="C36XTCgndYu3554YtBCTV73aV",
                        project_name="pstn-baselines", workspace="frederikwarburg")

    opt = TrainOptions().parse()
    experiment.log_parameters(vars(opt))
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

    if opt.visualize and opt.model.lower() in ['stn','pstn']:
        writer.visualize_transformation(model, 0)

    with experiment.train():

        for epoch in range(opt.epochs):

            model.train()
            epoch_start_time = time.time()
            iter_data_time = time.time()
            epoch_iter = 0
            ncorrect, nexamples = 0, 0

            for i, (input, label) in enumerate(dataset):
                input,label = input.to(device), label.to(device)
                iter_start_time = time.time()
                if total_steps % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time
                total_steps += opt.batch_size
                epoch_iter += opt.batch_size

                optimizer.zero_grad()
                output = model(input)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()

                if total_steps % opt.print_freq == 0:
                    t = (time.time() - iter_start_time) / opt.batch_size
                    writer.print_current_losses(epoch, epoch_iter, loss, t, t_data)
                    writer.plot_loss(loss, epoch, epoch_iter, dataset_size)

                    if opt.criterion.lower() == 'elbo':
                        print(criterion.nll, criterion.kl, criterion.rec)
                        writer.plot_loss_components(criterion.nll, criterion.kl, criterion.rec, epoch, epoch_iter, dataset_size)

                iter_data_time = time.time()

                predictions = evaluate(output, label)
                ncorrect, nexamples = ncorrect + predictions[0], nexamples + predictions[1]

            acc = ncorrect / nexamples
            experiment.log_metric("train_acc", 100 * acc, step=epoch)
            writer.plot_acc(acc, epoch, 'train')

            if epoch % opt.run_test_freq == 0:
                print('saving the model at the end of epoch %d, iters %d' %
                      (epoch, total_steps))

                # evaluate on test set
                acc = run_test(epoch, model)
                experiment.log_metric("test_acc", 100 * acc, step=epoch)
                writer.plot_acc(acc, epoch, 'test')

                is_best = acc > best_acc
                #save_network(model, opt, epoch, is_best)

                if opt.visualize and opt.model.lower() in ['stn','pstn']:
                    writer.visualize_transformation(model, epoch)

            print('End of epoch %d \t Time Taken: %d sec' %
                  (epoch, time.time() - epoch_start_time))
            scheduler.step()

    writer.close()
