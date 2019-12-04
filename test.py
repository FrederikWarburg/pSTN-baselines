from options.test_options import TestOptions
from data import DataLoader
from models import create_model
from utils.writer import Writer
import torch
from utils.evaluate import evaluate

def run_test(epoch=-1):
    print('Running Test')
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    dataset = DataLoader(opt)
    model = create_model(opt)
    model.eval()
    writer = Writer(opt)
    # test
    writer.reset_counter()
    with torch.no_grad():
        for i, (input, label) in enumerate(dataset):
            pred = model(input)

            ncorrect, nexamples = evaluate(pred, label)

            writer.update_counter(ncorrect, nexamples)

    writer.print_acc(epoch, writer.acc)
    return writer.acc


if __name__ == '__main__':
    run_test()
