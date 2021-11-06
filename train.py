import time
import torch
import copy
#from torch.utils.tensorboard import SummaryWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_model(model, dataloaders, criterion, optimiser, scheduler, num_epochs):
    best_acc = 0
    since = time.time()
    #Writer will output to ./runs/ directory by default
    #writer = SummaryWriter()

    for epoch in range(num_epochs):
        print('Epoch {}'.format(epoch))
        for phase in ['train', 'test']:

            epoch_loss = 0
            epoch_acc = 0
            
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            

            num_images = 0
            batch = 0
            for inputs, gt_labels in dataloaders[phase]:
                batch+=1
                inputs = inputs.to(device)
                gt_labels = gt_labels.to(device)
                
                optimiser.zero_grad()
                
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                
                _, preds = torch.max(outputs, dim=1)
                loss = criterion(outputs, gt_labels)
            
                if phase == 'train':
                    loss.backward()
                    optimiser.step()

                # epoch statistics
                num_images += inputs.shape[0]
                epoch_loss += loss.item()*inputs.shape[0]
                epoch_acc += torch.sum(preds==gt_labels).item()
                
                
            
            if phase == 'train':
                scheduler.step()

            epoch_loss /= num_images
            epoch_acc /= num_images

            print('{}. Loss: {:.4f}, Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_weights = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'checkpoint.pt')

            #if phase == 'train':
                #writer.add_scalar('TrainLoss', epoch_loss, epoch)
                #writer.add_scalar('TrainAcc', epoch_acc, epoch)
            #if phase == 'val':
                #writer.add_scalar('ValLoss', epoch_loss, epoch)
                #writer.add_scalar('ValAcc', epoch_acc, epoch)
                
            
            
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_weights)
    return model
