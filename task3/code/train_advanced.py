"""Resume small, monitored sessions; choose checkpoints on validation only."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT))
import argparse
import json
import random
import shutil
import time
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from task3.code.task3 import AlignedPlateSeq2Seq
from task3.code.refined_model import RefinedPlateSeq2Seq
from utils import (windows_device, safe_training_config, prepare_plate_cache, PlateDataset,
                   plate_loader, new_run, save_json, save_checkpoint, save_history,
                   file_hash, update_metadata, plate_train_step, batch_pause,
                   predict_plates, plate_metrics, evaluate_plates, plot_history)


def make_optimizer(model, config):
    groups = [
        {'params': model.cnn.parameters(), 'lr': config['cnn_learning_rate']},
        {'params': model.height_projection.parameters(), 'lr': config['projection_learning_rate']},
        {'params': list(model.decoder.parameters())+list(model.output.parameters()),
         'lr': config['learning_rate']}]
    if hasattr(model, 'refinement'):
        groups.append({'params':model.refinement.parameters(),'lr':config['refinement_learning_rate']})
        groups.append({'params':[model.column_positions],'lr':config['position_learning_rate']})
    return torch.optim.Adam(groups)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('task3/config/aligned.json'))
    parser.add_argument('--session-epochs', type=int, default=2)
    parser.add_argument('--soft-minutes', type=float, default=13)
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    config = safe_training_config(json.loads((ROOT/args.config).read_text(encoding='utf-8')))
    assert config['architecture'] in ['aligned','refined'] and config['batch_size']<=16
    assert 1<=args.session_epochs<=4 and 1<=args.soft_minutes<=14
    device = windows_device(config['seed'])
    frame, vocabulary, cache, identity = prepare_plate_cache(config['crop_mode'])
    data_id = identity+':'+config['crop_mode']
    datasets = {split: PlateDataset(frame, vocabulary, cache, split,
                augment=config['augment'] and split=='train') for split in ['train','validation']}
    val_loader = plate_loader(datasets['validation'], config)
    pointer = ROOT/'run/task3/development/active_run.json'
    pointer.parent.mkdir(parents=True, exist_ok=True)
    path = ROOT/args.run if args.run else None
    if path is None and pointer.exists():
        previous = ROOT/json.loads(pointer.read_text(encoding='utf-8'))['run']
        if json.loads((previous/'config.json').read_text(encoding='utf-8')) == config:
            path = previous
    if path is None:
        path = new_run('task3', config, data_id)
        save_json(path/'vocabulary.json', vocabulary)
        shutil.copy2(Path(__file__), path/'source/train_advanced.py')
        save_json(pointer, {'run': str(path.relative_to(ROOT))})
    else:
        assert json.loads((path/'config.json').read_text(encoding='utf-8')) == config
        assert json.loads((path/'metadata.json').read_text(encoding='utf-8'))['data_id']==data_id
        assert json.loads((path/'vocabulary.json').read_text(encoding='utf-8'))==vocabulary
        for name, source in [('utils.py',ROOT/'src/utils.py'), ('task3.py',ROOT/'task3/code/task3.py'),
                             ('train_advanced.py',Path(__file__)), ('refined_model.py',ROOT/'task3/code/refined_model.py')]:
            assert file_hash(path/'source'/name)==file_hash(source), 'Source changed since checkpoint: '+name
    print('Run:',path.relative_to(ROOT),flush=True)
    model_class = RefinedPlateSeq2Seq if config['architecture']=='refined' else AlignedPlateSeq2Seq
    model = model_class(len(vocabulary),config).to(device)
    optimizer = make_optimizer(model,config)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=2,factor=.5,min_lr=1e-6)
    epoch,offset,loss_sum,best_accuracy,best_cer,best_epoch,wait = 1,0,0.,-1.,float('inf'),0,0
    elapsed_before = 0.
    history = []
    last_file = path/'last.pt'
    if last_file.exists():
        state = torch.load(last_file,map_location='cpu',weights_only=True)
        model.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optimizer_state'])
        scheduler.load_state_dict(state['scheduler_state'])
        epoch,offset,loss_sum = state['epoch'],state['offset'],state['loss_sum']
        best_accuracy,best_cer,best_epoch,wait = state['best_accuracy'],state['best_cer'],state['best_epoch'],state['wait']
        elapsed_before = state['training_seconds']
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        random.setstate(state['python_rng'])
        if (path/'history.csv').exists():
            history=pd.read_csv(path/'history.csv').query('epoch < @epoch').to_dict('records')
        print('Resume epoch',epoch,'at training example',offset,flush=True)
    else:
        initial = ROOT/config['initial_checkpoint']
        initial_meta = json.loads((initial.parent/'metadata.json').read_text(encoding='utf-8'))
        assert initial_meta['data_id']==data_id
        assert json.loads((initial.parent/'vocabulary.json').read_text(encoding='utf-8'))==vocabulary
        saved=torch.load(initial,map_location='cpu',weights_only=True)
        missing,unexpected=model.load_state_dict(saved['model_state'],strict=False)
        new_layers=('height_projection.','refinement.','column_positions')
        assert not unexpected and all(name.startswith(new_layers) for name in missing)
        print('Warm start existing layers; new residual/position layers start at zero.' if config['architecture']=='refined'
              else 'Warm start CNN, GRU and classifier; new vertical projection starts as height average.',flush=True)
    print('Training / validation:',len(datasets['train']),len(datasets['validation']),flush=True)
    print('Parameters:',sum(p.numel() for p in model.parameters()),flush=True)
    start=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    completed=0
    stop_reason='session_epochs'

    def checkpoint_progress():
        save_checkpoint(last_file,{
            'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),
            'scheduler_state':scheduler.state_dict(),'config':config,'epoch':epoch,'offset':offset,
            'loss_sum':loss_sum,'best_accuracy':best_accuracy,'best_cer':best_cer,'best_epoch':best_epoch,
            'wait':wait,'training_seconds':elapsed_before+time.perf_counter()-start,
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
            'python_rng':random.getstate()})

    while epoch<=config['epochs'] and wait<config['patience'] and completed<args.session_epochs:
        epoch_start=time.perf_counter()
        order=torch.randperm(len(datasets['train']),generator=torch.Generator().manual_seed(config['seed']+1000*epoch)).tolist()
        remaining=Subset(datasets['train'],order[offset:])
        loader=DataLoader(remaining,batch_size=config['batch_size'],shuffle=False,num_workers=0,
                          generator=torch.Generator().manual_seed(config['seed']+epoch))
        model.train()
        for X,y in loader:
            loss=plate_train_step(model,optimizer,X.to(device),y.to(device))
            offset+=len(X)
            loss_sum+=loss*len(X)
            batch_pause(device)
            if offset%(config['batch_size']*250)==0:
                checkpoint_progress()
                print(f'Epoch {epoch}: {offset}/{len(order)} train examples; CE={loss_sum/offset:.4f}',flush=True)
            if time.perf_counter()-start>=args.soft_minutes*60:
                checkpoint_progress()
                stop_reason='session_time_budget'
                break
        if stop_reason=='session_time_budget':
            break
        # Save the trained weights BEFORE validation so an interruption never
        # discards the whole epoch. Validation can be repeated on resume.
        checkpoint_progress()
        actual,predicted,val_loss=predict_plates(model,val_loader,device)
        metrics=plate_metrics(actual,predicted,vocabulary)
        accuracy,cer=metrics['plate_accuracy'],metrics['CER']
        better=accuracy>best_accuracy+1e-8 or (abs(accuracy-best_accuracy)<1e-8 and cer<best_cer-1e-8)
        history.append({'epoch':epoch,'train_loss':loss_sum/len(datasets['train']),
                        'val_loss':val_loss,'val_plate_accuracy':accuracy,'val_CER':cer,
                        'val_character_accuracy':metrics['character_accuracy'],
                        'lr':optimizer.param_groups[2]['lr'],'seconds':time.perf_counter()-epoch_start})
        save_history(path/'history.csv',history)
        if better:
            best_accuracy,best_cer,best_epoch,wait=accuracy,cer,epoch,0
            save_checkpoint(path/'best.pt',{'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),
                'config':config,'epoch':epoch,'validation_score':accuracy,'validation_CER':cer})
            update_metadata(path,parameters=sum(p.numel() for p in model.parameters()),
                            training_seconds=elapsed_before+time.perf_counter()-start,
                            best_epoch=best_epoch,validation_score=best_accuracy,validation_CER=best_cer)
            evaluate_plates(model,val_loader,frame,vocabulary,path,device,'validation',
                            predictions=(actual,predicted,val_loss))
        else:
            wait+=1
        scheduler.step(val_loss)
        print(f'Epoch {epoch}: val_plate={accuracy:.4%}, char={metrics["character_accuracy"]:.4%}, CER={cer:.4%}',flush=True)
        epoch+=1
        offset,loss_sum=0,0.
        completed+=1
        checkpoint_progress()
        update_metadata(path,training_seconds=elapsed_before+time.perf_counter()-start,
                        parameters=sum(p.numel() for p in model.parameters()),best_epoch=best_epoch,
                        validation_score=best_accuracy,validation_CER=best_cer,
                        peak_vram_gib=torch.cuda.max_memory_allocated()/2**30,
                        last_completed_epoch=epoch-1,training_complete=False)
        plot_history(path,'val_plate_accuracy')
        if time.perf_counter()-start>=args.soft_minutes*60:
            stop_reason='session_time_budget'
            break
    finished=epoch>config['epochs'] or wait>=config['patience']
    if finished:
        stop_reason='max_epochs' if epoch>config['epochs'] else 'early_stopping'
    checkpoint_progress()
    update_metadata(path,session_status='complete',training_complete=finished,stop_reason=stop_reason,
                    training_seconds=elapsed_before+time.perf_counter()-start)
    sessions=path/'sessions'
    sessions.mkdir(exist_ok=True)
    save_json(sessions/(datetime.now().strftime('%Y%m%d_%H%M%S')+'.json'),{
        'completed_epochs':completed,'next_epoch':epoch,'next_training_offset':offset,
        'elapsed_seconds':time.perf_counter()-start,'reason':stop_reason,'training_complete':finished})
    print('Session finished normally:',stop_reason,'; training complete:',finished,flush=True)


if __name__=='__main__':
    main()
