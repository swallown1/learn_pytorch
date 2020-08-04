import torch
import torch.utils.data as Data

batch_size = 4

x = torch.linspace(1,10,20)
y = torch.linspace(10,1,20)

# 先转换成 torch 能识别的 Dataset
torch_data = Data.TensorDataset(x, y)

# 把 dataset 放入 DataLoader
loader = Data.DataLoader(
    dataset=torch_data,
    batch_size=batch_size,
    shuffle=True,
    # num_workers=2,
)

for epoch in range(3):
    for step,(batch_x,batch_y) in enumerate(loader):
        # 打出来一些数据
        print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',batch_x.numpy(), '| batch y: ', batch_y.numpy())