import torch
import numpy as np
import torchvision
#from . import measure
import torchvision.transforms as transforms
import torchvision.models as models
#resnet18 = models.resnet18()




def get_batch_jacobian(net, x, target):
    net.zero_grad()

    x.requires_grad_(True)

    _, y = net(x)

    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()

    return jacob, target.detach()


def eval_score(jacob, labels=None):
    corrs = np.corrcoef(jacob)
    v, _ = np.linalg.eig(corrs)
    k = 1e-5
    return -np.sum(np.log(v + k) + 1.0 / (v + k))

'''    添加BN层
@measure("jacov", bn=True)
def compute_jacob_cov(net, inputs, targets, split_data=1, loss_fn=None):
    device = inputs.device
    # Compute gradients (but don't apply them)
    net.zero_grad()

    jacobs, labels = get_batch_jacobian(
        net, inputs, targets, device, split_data=split_data
    )
    jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()

    try:
        jc = eval_score(jacobs, labels)
    except Exception as e:
        print(e)
        jc = np.nan

    return jc
'''
train_data = torchvision.datasets.CIFAR10(root='/home/zsl/SR/zero-cost/dataset',
                                             train=True,
                                             transform=transforms.ToTensor(),
                                             )
train_loader = torch.utils.data.DataLoader(train_data, batch_size=128,
                                                       num_workers=0, pin_memory=pin_memory, sampler= RepeatSampler(torch.utils.data.sampler.SubsetRandomSampler(train_split), repeat))


data_iterator = iter(train_loader)
x, target = next(data_iterator)

resnet18 = models.resnet18()
resnet34 = models.resnet34()
resnet50 = models.resnet50()

jacobs, labels = get_batch_jacobian(resnet18, x, target)
# print('done get jacobs')
jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()


try:
    score = eval_score(jacobs, labels)
    print(score)
except Exception as e:
    print(e)
    score = -10e8
    
