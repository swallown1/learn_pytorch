import torch

x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)

y = x.pow(2) + 0.2*torch.rand(x.size())

net = torch.nn.Sequential(
    torch.nn.Linear(1,10),
    torch.nn.ReLU(),
    torch.nn.Linear(10,1)
)

optimizer = torch.optim.SGD(net.parameters(),lr=0.01)
loss_fun = torch.nn.MSELoss()

for i in range(100):
    pred = net(x)
    loss= loss_fun(pred,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(str(i)+": loss "+str(loss))

# torch.save(net,'net.pkl')
torch.save(net.state_dict(),'net_param.pkl')


def restore_net(x):
    net2 = torch.load('net.pkl')
    pred = net2(x)
    return pred

# print(restore_net(x),y)
