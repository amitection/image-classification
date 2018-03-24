import torch.nn as nn
import pdb

class Demo_Model(nn.Module):
    def __init__(self, nClasses=200):
        super(Demo_Model, self).__init__();

        self.conv_1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.relu_1 = nn.ReLU(True);
        self.batch_norm_1 = nn.BatchNorm2d(32);
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_2 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.relu_2 = nn.ReLU(True);
        self.batch_norm_2 = nn.BatchNorm2d(32);
        '''self.pool_2 = nn.MaxPool2d(kernel_size = 2, stride =2)

        self.conv_3 = nn.Conv2d(32,32,kernel_size=5,stride=1, padding = 2)
        self.relu_3 = nn.ReLU(True);
        self.batch_norm_3 = nn.BatchNorm2d(32);'''

        self.fc_1 = nn.Linear(32768, 1024);
        # self.fc_1 = nn.Linear(8192, 200);
        self.relu_4 = nn.ReLU(True);
        self.batch_norm_4 = nn.BatchNorm1d(1024);
        self.dropout_1 = nn.Dropout(p=0.5);
        self.fc_2 = nn.Linear(1024, nClasses);

    def forward(self, x):
        # pdb.set_trace();
        y = self.conv_1(x)
        y = self.relu_1(y)
        y = self.batch_norm_1(y)
        y = self.pool_1(y)

        y = self.conv_2(y)
        y = self.relu_2(y)
        y = self.batch_norm_2(y)
        '''y = self.pool_2(y)

        y = self.conv_3(y)
        y = self.relu_3(y)
        y = self.batch_norm_3(y)'''

        y = y.view(y.size(0), -1)
        y = self.fc_1(y)
        y = self.relu_4(y)
        y = self.batch_norm_4(y)
        y = self.dropout_1(y)
        y = self.fc_2(y)
        return (y)
