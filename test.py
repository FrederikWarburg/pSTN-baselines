from options.test_options import TestOptions
from data import DataLoader
from models import create_model
from utils.writer import Writer
import torch
from utils.evaluate import evaluate

def run_test(epoch=-1, model = None):
    print('Running Test')
    opt = TestOptions().parse()
    opt.shuffle = False  # no shuffle
    dataset = DataLoader(opt)

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

    writer.print_acc(epoch, writer.acc)
    return writer.acc


if __name__ == '__main__':
    run_test()
