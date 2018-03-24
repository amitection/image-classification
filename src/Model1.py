import torch.nn as nn

class Model1(nn.Module):
    def __init__(self, nClasses=200):

        # [CONV RELU POOL] x N
        super(Model1, self).__init__();

        self.conv_1 = nn.Conv2d(3, 128, kernel_size=5, stride=1, padding=2)
        self.relu_1 = nn.ReLU(True);
        self.batch_norm_1 = nn.BatchNorm2d(128);
        output = ((56 - 5 + (2 * 2)) / 1) + 1
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        output = ((output - 2 + (2 * 0)) / 2) + 1


        self.conv_2 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.relu_2 = nn.ReLU(True);
        self.batch_norm_2 = nn.BatchNorm2d(128);
        output = ((output - 5 + (2 * 2)) / 1) + 1
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        output = ((output - 2 + (2 * 0)) / 2) + 1

        self.conv_3 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.relu_3 = nn.ReLU(True);
        self.batch_norm_3 = nn.BatchNorm2d(128);
        output = ((output - 5 + (2 * 2)) / 1) + 1
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        output = ((output - 2 + (2 * 0)) / 2) + 1

        self.conv_4 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.relu_4 = nn.ReLU(True);
        self.batch_norm_4 = nn.BatchNorm2d(128);
        output = ((output - 5 + (2 * 2)) / 1) + 1
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        output = ((output - 2 + (2 * 0)) / 2) + 1

        # CONV
        self.conv_5 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.relu_5 = nn.ReLU(True);
        output = ((output - 5 + (2 * 2)) / 1) + 1
        self.batch_norm_5 = nn.BatchNorm2d(128);

        output = output * output * 128
        print "ouput:  "+str(output)


        # Affine x M
        self.fc_1 = nn.Linear(output, 1024);
        self.relu_6 = nn.Softmax(1);
        self.batch_norm_6 = nn.BatchNorm1d(1024);
        self.dropout_1 = nn.Dropout(p=0.5);

        # Softmax or SVM
        self.fc_2 = nn.Linear(1024, nClasses);

    def forward(self, x):

        y = self.conv_1(x)
        y = self.relu_1(y)
        y = self.batch_norm_1(y)
        y = self.pool_1(y)

        y = self.conv_2(y)
        y = self.relu_2(y)
        y = self.batch_norm_2(y)
        y = self.pool_2(y)

        y = self.conv_3(y)
        y = self.relu_3(y)
        y = self.batch_norm_3(y)
        y = self.pool_3(y)

        y = self.conv_4(y)
        y = self.relu_4(y)
        y = self.batch_norm_4(y)
        y = self.pool_4(y)

        y = self.conv_5(y)
        y = self.relu_5(y)
        y = self.batch_norm_5(y)

        y = y.view(y.size(0), -1)
        y = self.fc_1(y)
        y = self.relu_6(y)
        y = self.batch_norm_6(y)
        y = self.dropout_1(y)
        y = self.fc_2(y)
        return (y)