import torch.nn as nn

class My_Model(nn.Module):
    def __init__(self, nClasses=200):


        super(My_Model, self).__init__();

        # CONV 1
        self.conv_1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=2)
        self.relu_1 = nn.LeakyReLU(0.1);
        self.batch_norm_1 = nn.BatchNorm2d(128);
        output = ((258 - 3 + (2 * 2)) / 1) + 1
        # Pool 1
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        output = ((output - 2 + (2 * 0)) / 2) + 1

        # CONV 2
        self.conv_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2)
        self.relu_2 = nn.LeakyReLU(0.1);
        self.batch_norm_2 = nn.BatchNorm2d(128);
        output = ((output - 3 + (2 * 2)) / 1) + 1
        # Pool 2
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        output = ((output - 2 + (2 * 0)) / 2) + 1

        # CONV 3
        self.conv_3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2)
        self.relu_3 = nn.LeakyReLU(0.1);
        self.batch_norm_3 = nn.BatchNorm2d(128);
        output = ((output - 3 + (2 * 2)) / 1) + 1
        # Pool 3
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        output = ((output - 2 + (2 * 0)) / 2) + 1

        # CONV 4
        self.conv_4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2)
        self.relu_4 = nn.LeakyReLU(0.1);
        self.batch_norm_4 = nn.BatchNorm2d(128);
        output = ((output - 3 + (2 * 2)) / 1) + 1
        # Pool 4
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        output = ((output - 2 + (2 * 0)) / 2) + 1

        # CONV 5
        self.conv_5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2)
        self.relu_5 = nn.LeakyReLU(0.1);
        self.batch_norm_5 = nn.BatchNorm2d(128);
        output = ((output - 3 + (2 * 2)) / 1) + 1
        # Pool 5
        self.pool_5 = nn.MaxPool2d(kernel_size=2, stride=2)
        output = ((output - 2 + (2 * 0)) / 2) + 1

        # CONV 6
        self.conv_6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2)
        self.relu_6 = nn.LeakyReLU(0.1);
        self.batch_norm_6 = nn.BatchNorm2d(128);
        output = ((output - 3 + (2 * 2)) / 1) + 1
        # Pool 6
        self.pool_6 = nn.MaxPool2d(kernel_size=2, stride=2)
        output = ((output - 2 + (2 * 0)) / 2) + 1

        # CONV 7
        self.conv_7 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2)
        self.relu_7 = nn.LeakyReLU(0.1);
        output = ((output - 3 + (2 * 2)) / 1) + 1
        self.batch_norm_7 = nn.BatchNorm2d(128);

        output = output * output * 128
        print "ouput:  "+str(output)


        # Affine x M
        self.fc_1 = nn.Linear(output, 1024);
        # nn.init.xavier_uniform(self.fc_1.weight);
        self.relu_8 = nn.Softmax(1);
        self.batch_norm_8 = nn.BatchNorm1d(1024);
        self.dropout_1 = nn.Dropout(p=0.5);

        # self.fc_2 = nn.Linear(1024, 1024, bias=True);
        # nn.init.xavier_uniform(self.fc_2.weight);
        # self.relu_9 = nn.Softmax(1);
        # self.batch_norm_9 = nn.BatchNorm1d(1024);
        # self.dropout_2 = nn.Dropout(p=0.5);

        # Softmax or SVM
        self.fc_3 = nn.Linear(1024, nClasses);
        # nn.init.xavier_uniform(self.fc_3.weight)

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
        y = self.pool_5(y)

        y = self.conv_6(y)
        y = self.relu_6(y)
        y = self.batch_norm_6(y)
        y = self.pool_6(y)

        y = self.conv_7(y)
        y = self.relu_7(y)
        y = self.batch_norm_7(y)

        y = y.view(y.size(0), -1)
        y = self.fc_1(y)
        y = self.relu_8(y)
        y = self.batch_norm_8(y)
        y = self.dropout_1(y)

        # y = self.fc_2(y)
        # y = self.relu_9(y)
        # y = self.batch_norm_9(y)
        # y = self.dropout_2(y)

        y = self.fc_3(y)
        return (y)
