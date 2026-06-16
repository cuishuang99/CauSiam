import torch as T
import wandb
wandb_init = wandb.init(project='LR',
                        id=wandb.util.generate_id(),
                        name='lr',)

epoch = 20
resume_epoch = 10

cc = T.nn.Conv2d(10,10,3)
optimizer = T.optim.SGD(cc.parameters(), lr=0.1)

state = T.load('state.pt')
optimizer.load_state_dict(state['optimizer'])
scheduler = T.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch,eta_min=1e-6)
# scheduler.load_state_dict(state['scheduler'])

for i in range(resume_epoch,epoch):
    
    lr1 = optimizer.param_groups[0]["lr"]
    lr = scheduler.get_last_lr()[0]
    wandb.log({'epoch':i,
               'lr':lr,
               'lr1':lr1})
    optimizer.step()
    scheduler.step()
    # if i == resume_epoch - 1:
    #     state = {'optimizer':optimizer.state_dict(),
    #                 'scheduler':scheduler.state_dict(),}
    #     T.save(state,'state.pt')