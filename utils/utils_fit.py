import os

import torch
from nets.centernet_training import focal_loss, reg_l1_loss
from tqdm import tqdm

from utils.utils import get_lr


def fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, backbone, save_period, save_dir):
    total_r_loss    = 0
    total_c_loss    = 0
    total_loss      = 0
    val_loss        = 0

    model_train.train()
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break

            with torch.no_grad():
                if cuda:
                    batch = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in batch]
                else:
                    batch = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in batch]
            batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks = batch

            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()

            if backbone=="resnet50" or backbone=="resnet50dcn":
                hm, wh, offset  = model_train(batch_images)
                c_loss          = focal_loss(hm, batch_hms)
                wh_loss         = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
                off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)
                
                loss            = c_loss + wh_loss + off_loss

                total_loss      += loss.item()
                total_c_loss    += c_loss.item()
                total_r_loss    += wh_loss.item() + off_loss.item()
            elif backbone=="resnet18" or backbone=="resnet18dcn":
                hm, wh, offset  = model_train(batch_images)
                c_loss          = focal_loss(hm, batch_hms)
                wh_loss         = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
                off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)
                
                loss            = c_loss + wh_loss + off_loss

                total_loss      += loss.item()
                total_c_loss    += c_loss.item()
                total_r_loss    += wh_loss.item() + off_loss.item()
            elif backbone == "mobilenetv2" or backbone == "mobilenetv3" or backbone == "ghostnet" or backbone=="shufflenet" :
                hm, wh, offset  = model_train(batch_images)
                c_loss          = focal_loss(hm, batch_hms)
                wh_loss         = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
                off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)
                
                loss            = c_loss + wh_loss + off_loss

                total_loss      += loss.item()
                total_c_loss    += c_loss.item()
                total_r_loss    += wh_loss.item() + off_loss.item()
            
            elif backbone =="mobilenetv2_mo" or backbone=="mobilenetv3_mo" or backbone=="pgmdet" or  backbone=="mobilenetv2_centernet" or backbone=="pgmdet3" or backbone=="pgmdet4" or backbone=="pgmdet5" or backbone=="pgmdet6":
                hm, wh, offset  = model_train(batch_images)
                c_loss          = focal_loss(hm, batch_hms)
                wh_loss         = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
                off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)
                
                loss            = c_loss + wh_loss + off_loss

                total_loss      += loss.item()
                total_c_loss    += c_loss.item()
                total_r_loss    += wh_loss.item() + off_loss.item()

            else:
                outputs         = model_train(batch_images)
                loss            = 0
                c_loss_all      = 0
                r_loss_all      = 0
                index           = 0
                for output in outputs:
                    hm, wh, offset = output["hm"].sigmoid(), output["wh"], output["reg"]
                    c_loss      = focal_loss(hm, batch_hms)
                    wh_loss     = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
                    off_loss    = reg_l1_loss(offset, batch_regs, batch_reg_masks)

                    loss        += c_loss + wh_loss + off_loss
                    
                    c_loss_all  += c_loss
                    r_loss_all  += wh_loss + off_loss
                    index       += 1
                total_loss      += loss.item() / index
                total_c_loss    += c_loss_all.item() / index
                total_r_loss    += r_loss_all.item() / index
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix(**{'total_r_loss'  : total_r_loss / (iteration + 1), 
                                'total_c_loss'  : total_c_loss / (iteration + 1),
                                'lr'            : get_lr(optimizer)})
            pbar.update(1)


    print('Finish Train')

    model_train.eval()
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    batch = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in batch]
                else:
                    batch = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in batch]
                batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks = batch

                if backbone=="resnet50" or backbone=="resnet50dcn":
                    hm, wh, offset  = model_train(batch_images)
                    c_loss          = focal_loss(hm, batch_hms)
                    wh_loss         = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
                    off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)

                    loss            = c_loss + wh_loss + off_loss

                    val_loss        += loss.item()
                elif backbone=="resnet18" or backbone=="resnet18dcn":
                    hm, wh, offset  = model_train(batch_images)
                    c_loss          = focal_loss(hm, batch_hms)
                    wh_loss         = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
                    off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)

                    loss            = c_loss + wh_loss + off_loss

                    val_loss        += loss.item()
                elif backbone=="mobilenetv2" or backbone=="mobilenetv3" or backbone=="ghostnet" or backbone=="shufflenet":
                    hm, wh, offset  = model_train(batch_images)
                    c_loss          = focal_loss(hm, batch_hms)
                    wh_loss         = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
                    off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)

                    loss            = c_loss + wh_loss + off_loss

                    val_loss        += loss.item()
                
                elif backbone=="mobilenetv2_mo" or backbone=="mobilenetv3_mo" or backbone=="pgmdet" or backbone=="mobilenetv2_centernet" or backbone=="pgmdet3" or backbone=="pgmdet4" or backbone=="pgmdet5" or backbone=="pgmdet6":
                    hm, wh, offset  = model_train(batch_images)
                    c_loss          = focal_loss(hm, batch_hms)
                    wh_loss         = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
                    off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)

                    loss            = c_loss + wh_loss + off_loss

                    val_loss        += loss.item()


                else:
                    outputs = model_train(batch_images)
                    index = 0
                    loss = 0
                    for output in outputs:
                        hm, wh, offset  = output["hm"].sigmoid(), output["wh"], output["reg"]
                        c_loss          = focal_loss(hm, batch_hms)
                        wh_loss         = 0.1*reg_l1_loss(wh, batch_whs, batch_reg_masks)
                        off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)

                        loss            += c_loss + wh_loss + off_loss
                        index           += 1
                    val_loss            += loss.item() / index

                pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
                pbar.update(1)
    print('Finish Validation')
    
    loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))
