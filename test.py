from options.test_options import TestOptions
from data import DataLoader
from models import create_model
from utils.writer import Writer
import torch
from utils.evaluate import evaluate

def run_test(epoch=-1, model = None, training_data = False):
    print('Running Test')
    opt = TestOptions().parse()
    opt.no_shuffle = True  # no shuffle
    opt.is_train = training_data
    dataset = DataLoader(opt)
    opt.is_train = False

    if model == None:
        print("create new model")
        model = create_model(opt)

    writer = Writer(opt)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # test
    writer.reset_counter()
    model.eval()
    with torch.no_grad():
        for i, (input, label) in enumerate(dataset):
            input,label = input.to(device), label.to(device)
            pred = model(input)

            ncorrect, nexamples = evaluate(pred, label)

            writer.update_counter(ncorrect, nexamples)

    mode = 'train' if training_data else 'test'
    writer.print_acc(epoch, writer.acc, mode)
    return writer.acc


if __name__ == '__main__':
    run_test()
