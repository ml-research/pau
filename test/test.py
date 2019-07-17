import torch
from pau.torchsummary import summary
from pau.utils import PAU


def main():
    batch_size = 8
    D_in = 2
    H = 4
    D_out = 2

    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        PAU(),  # e.g. instead of torch.nn.ReLU()
        torch.nn.Linear(H, D_out),
    )

    model.cuda()
    summary(model, input_size=(1, D_in))

    input_tensor = torch.rand((batch_size, D_in)).cuda()
    output_tensor = model(input_tensor)


if __name__ == '__main__':
    main()
