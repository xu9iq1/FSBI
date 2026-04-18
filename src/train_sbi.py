import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from pathlib import Path
from PIL import Image
import sys
import random
from utils.sbi import SBI_Dataset
from utils.esbi import ESBI_Dataset
from utils.scheduler import LinearDecayLR
from sklearn.metrics import confusion_matrix, roc_auc_score
import argparse
from utils.logs import log
from utils.funcs import load_json
from utils.runtime import get_device, seed_everything
from datetime import datetime
from tqdm import tqdm
from model import Detector


def compute_accuray(pred,true):
    pred_idx=pred.argmax(dim=1).cpu().data.numpy()
    tmp=pred_idx==true.cpu().numpy()
    return sum(tmp)/len(pred_idx)


def parse_saved_auc(path):
    try:
        return float(Path(path).stem.split("_")[1])
    except (IndexError, ValueError):
        return None


def main(args):
    cfg=load_json(args.config)
    if args.epoch is not None:
        cfg['epoch'] = int(args.epoch)

    seed = args.seed
    device = get_device(args.device)
    seed_everything(seed, device)
    print(f'Using device: {device}')

    image_size=cfg['image_size']
    batch_size=cfg['batch_size']

    if args.variant == 'sbi':
        train_dataset = SBI_Dataset(phase='train', image_size=image_size)
        val_dataset = SBI_Dataset(phase='val', image_size=image_size)
    else:
        train_dataset = ESBI_Dataset(
            phase='train',
            image_size=image_size,
            wavelet=args.wavelet,
            mode=args.mode,
        )
        val_dataset = ESBI_Dataset(
            phase='val',
            image_size=image_size,
            wavelet=args.wavelet,
            mode=args.mode,
        )

    pin_memory = device.type == 'cuda'
    train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size//2,shuffle=True,collate_fn=train_dataset.collate_fn,num_workers=args.num_workers,pin_memory=pin_memory,drop_last=True,worker_init_fn=train_dataset.worker_init_fn)
    val_loader=torch.utils.data.DataLoader(val_dataset,batch_size=batch_size,shuffle=False,collate_fn=val_dataset.collate_fn,num_workers=args.num_workers,pin_memory=pin_memory,worker_init_fn=val_dataset.worker_init_fn)
    
    model=Detector()
    # cnn_sd=torch.load("/home/g202302610/Documents/SelfBlendedImages/output/epoch_splits/eb4_dwtsym2reflect_sbi_base_01_07_09_01_51/weights/75_0.9980_val.tar")["model"]
    # model.load_state_dict(cnn_sd)
    model=model.to(device)

    iter_loss=[]
    train_losses=[]
    test_losses=[]
    train_accs=[]
    test_accs=[]
    val_accs=[]
    val_losses=[]
    n_epoch=cfg['epoch']
    lr_scheduler=LinearDecayLR(model.optimizer, n_epoch, int(n_epoch*0.75))
    last_loss=99999

    now=datetime.now()
    session_name = args.session_name or 'run'
    start_epoch = 0
    if args.resume:
        resume_path = Path(args.resume)
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        model.optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        lr_scheduler.last_epoch = start_epoch - 1
        save_path = str(
            (Path(args.output_dir) if args.output_dir else resume_path.parent.parent).resolve()
        ) + "/"
        os.makedirs(save_path + 'weights/', exist_ok=True)
        os.makedirs(save_path + 'logs/', exist_ok=True)
        print(f"Resuming from {resume_path} at epoch {start_epoch + 1}")
    else:
        save_path = args.output_dir
        if save_path is None:
            save_path='output/{}_'.format(session_name)+now.strftime(os.path.splitext(os.path.basename(args.config))[0])+'_'+now.strftime("%m_%d_%H_%M_%S")+'/'
        if not save_path.endswith('/'):
            save_path += '/'
        os.makedirs(save_path+'weights/', exist_ok=False)
        os.makedirs(save_path+'logs/', exist_ok=False)
    logger = log(path=save_path+"logs/", file="losses.logs")

    criterion=nn.CrossEntropyLoss()


    last_auc, last_val_auc=0, 0
    weight_dict={}
    n_weight=5
    for weight_path in Path(save_path, "weights").glob("*.tar"):
        auc = parse_saved_auc(weight_path)
        if auc is not None:
            weight_dict[str(weight_path)] = auc
    if weight_dict:
        last_val_auc=min([weight_dict[k] for k in weight_dict])

    for epoch in range(start_epoch, n_epoch):
        np.random.seed(seed + epoch)
        train_loss=0.
        train_acc=0.
        model.train()
        for step,data in enumerate(tqdm(train_loader)):
            img=data['img'].to(device, non_blocking=pin_memory).float()
            target=data['label'].to(device, non_blocking=pin_memory).long()
            output=model.training_step(img, target)
            loss=criterion(output,target)
            loss_value=loss.item()
            iter_loss.append(loss_value)
            train_loss+=loss_value
            acc=compute_accuray(F.log_softmax(output,dim=1),target)
            train_acc+=acc
        train_losses.append(train_loss/len(train_loader))
        train_accs.append(train_acc/len(train_loader))

        log_text="Epoch {}/{} | train loss: {:.4f}, train acc: {:.4f}, ".format(epoch+1,n_epoch,train_loss/len(train_loader),train_acc/len(train_loader),)
        lr_scheduler.step()

        model.eval()
        val_acc, val_loss=0.,0.
        output_dict, target_dict=[],[]
        np.random.seed(seed)
        for step,data in enumerate(tqdm(val_loader)):
            img=data['img'].to(device, non_blocking=pin_memory).float()
            target=data['label'].to(device, non_blocking=pin_memory).long()
            
            with torch.no_grad():
                output=model(img)
                loss=criterion(output,target)
            
            loss_value=loss.item()
            iter_loss.append(loss_value)
            val_loss+=loss_value
            acc=compute_accuray(F.log_softmax(output,dim=1),target)
            val_acc+=acc
            output_dict+=output.softmax(1)[:,1].cpu().data.numpy().tolist()
            target_dict+=target.cpu().data.numpy().tolist()
            
        val_losses.append(val_loss/len(val_loader))
        val_accs.append(val_acc/len(val_loader))
        val_auc=roc_auc_score(target_dict,output_dict)
        log_text+="val loss: {:.4f}, val acc: {:.4f}, val auc: {:.4f}".format(val_loss/len(val_loader),val_acc/len(val_loader),val_auc)

        # if epoch+1 == 75:
        #     save_model_path=os.path.join(save_path+'weights/',"{}_{:.4f}_val.tar".format(epoch+1,val_auc))
        #     torch.save({"model":model.state_dict(),"optimizer":model.optimizer.state_dict(),"epoch":epoch},save_model_path)

        if len(weight_dict)<n_weight:
            save_model_path=os.path.join(save_path+'weights/',"{}_{:.4f}_val.tar".format(epoch+1,val_auc))
            weight_dict[save_model_path]=val_auc
            torch.save({"model":model.state_dict(),"optimizer":model.optimizer.state_dict(),"epoch":epoch},save_model_path)
            last_val_auc=min([weight_dict[k] for k in weight_dict])

        elif val_auc>=last_val_auc:
            save_model_path=os.path.join(save_path+'weights/',"{}_{:.4f}_val.tar".format(epoch+1,val_auc))
            for k in weight_dict:
                if weight_dict[k]==last_val_auc:
                    del weight_dict[k]
                    os.remove(k)
                    weight_dict[save_model_path]=val_auc
                    break
            torch.save({"model":model.state_dict(),"optimizer":model.optimizer.state_dict(),"epoch":epoch},save_model_path)
            last_val_auc=min([weight_dict[k] for k in weight_dict])
        
        logger.info(log_text)
        print(lr_scheduler.get_lr())
        print()

        
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument(dest='config')
    parser.add_argument('-n',dest='session_name')
    parser.add_argument('-w',dest='wavelet')
    parser.add_argument('-m',dest='mode')
    parser.add_argument('-e',dest='epoch')
    parser.add_argument('--resume')
    parser.add_argument('--output-dir')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--variant', choices=['esbi', 'sbi'], default='esbi')
    args=parser.parse_args()
    main(args)
        
