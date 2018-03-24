from torch.optim.lr_scheduler import ReduceLROnPlateau

class ExtReduceLROnPlateau(ReduceLROnPlateau):

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 verbose=False, threshold=1e-4, threshold_mode='rel',
                 cooldown=0, min_lr=0, eps=1e-8):
        #import ipdb; ipdb.set_trace()
        super(ExtReduceLROnPlateau, self).__init__(optimizer, mode, factor, patience, verbose, threshold, threshold_mode,
              cooldown, min_lr, eps)

    def getLr(self):
        #return self.optimizer.param_groups[0]['lr']
        for i, param_group in enumerate(self.optimizer.param_groups):
            print("####################### " + str(i) + " -- " + str(param_group['lr']))
            lr = param_group['lr']

        return lr