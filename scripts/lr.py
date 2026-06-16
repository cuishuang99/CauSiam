import torch as T
import wandb
wandb_init = wandb.init(project='LR',
                        id=wandb.util.generate_id(),
                        name='lr',)
epoch = 6
milestones=[int(i*epoch/3) for i in range(1,3)]
cc = T.nn.Conv2d(10,10,3)
optimizer = T.optim.SGD(cc.parameters(), lr=0.1)
scheduler = T.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones,gamma=0.5)

for i in range(epoch):
    lr1 = optimizer.param_groups[0]["lr"]
    lr = scheduler.get_last_lr()[0]
    # wandb.log({'epoch':i,
    #            'lr':lr,
    #            'lr1':lr1})
    print(lr,lr1)
    optimizer.step()
    scheduler.step()
    
state = {'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),}
# T.save(state,'state.pt')