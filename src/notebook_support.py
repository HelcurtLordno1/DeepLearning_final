"""Read saved experiments and keep presentation notebooks free of file management."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from utils import ROOT


def setup_plots():
    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib', 'inline')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.dpi': 100, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.grid': True,
                         'grid.alpha': .18, 'font.size': 10})
    pd.set_option('display.max_columns', 12)
    pd.set_option('display.max_colwidth', 36)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def plate_runs():
    rows = []
    for path in sorted((ROOT/'run/task3').glob('*/*/*/metadata.json')):
        meta = read_json(path)
        if not meta.get('validation_evaluated'):
            continue
        config = read_json(path.parent/'config.json')
        rows.append({'model': config['name'], 'seed': config['seed'],
                     'plate_accuracy': meta['validation_score'],
                     'CER': meta['validation_CER'], 'path': path.parent})
    return rows


def load_run(task, folder=None, architecture='legacy'):
    if folder is not None:
        folder = Path(folder)
        split = 'validation'
    elif task == 'task3':
        selection = ROOT/'run/task3/final_selection.json'
        selected = read_json(selection) if selection.exists() else None
        candidates = [row for row in plate_runs() if
                      read_json(row['path']/'config.json').get('architecture','legacy') == architecture]
        folder = max(candidates, key=lambda row: (row['plate_accuracy'], -row['CER']))['path']
        if selected:
            chosen = ROOT/selected['selected_run']
            if read_json(chosen/'config.json').get('architecture','legacy') == architecture:
                folder = chosen
        split = 'test' if selected and folder == ROOT/selected['selected_run'] and (folder/'test_metrics.json').exists() else 'validation'
    else:
        folder = ROOT/read_json(ROOT/'run/selected_configs.json')[task]['seed42_run']
        split = 'test'
    config = read_json(folder/'config.json')
    metadata = read_json(folder/'metadata.json')
    metric_file = folder/(split+'_metrics.json')
    recovered = False
    if task == 'task3':
        try:
            scores = read_json(metric_file)
        except json.JSONDecodeError:
            replacement = read_json(ROOT/'run/recovery/recovered_files.json')[str(metric_file.relative_to(ROOT))]
            scores = read_json(ROOT/replacement)
            recovered = True
    else:
        scores = pd.read_csv(folder/(split+'_metrics.csv'))
    return {'config': config, 'metadata': metadata, 'history': pd.read_csv(folder/'history.csv'),
            'scores': scores, 'predictions': pd.read_csv(folder/(split+'_predictions.csv')),
            'split': split, 'recovered_metrics': recovered, '_folder': folder}


def load_weights(model, run):
    saved = torch.load(run['_folder']/'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(saved['model_state'])
    model.eval()


def layer_shapes(model, *inputs):
    rows, hooks = [], []
    def record(name):
        def hook(module, args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            rows.append({'Layer': name+' ('+module.__class__.__name__+')',
                         'Input': str(tuple(args[0].shape)), 'Output': str(tuple(tensor.shape)),
                         'Parameters': sum(p.numel() for p in module.parameters(recurse=False))})
        return hook
    for name, module in model.named_modules():
        if name and not list(module.children()):
            hooks.append(module.register_forward_hook(record(name)))
    try:
        with torch.no_grad():
            prediction = model(*inputs)
    finally:
        for hook in hooks:
            hook.remove()
    return pd.DataFrame(rows), prediction


def training_settings(run):
    config, meta = run['config'], run['metadata']
    table = pd.DataFrame({'Thông số': ['Seed', 'Hidden size', 'Batch run này', 'Learning rate ban đầu',
                                     'Epoch tối đa', 'Early-stopping patience', 'Epoch đã chạy',
                                     'Epoch checkpoint tốt nhất', 'Thời gian train (giây)', 'Số tham số'],
                         'Giá trị': [config['seed'], config['hidden_size'], config['batch_size'],
                                     config['learning_rate'], config['epochs'], config['patience'],
                                     len(run['history']), meta['best_epoch'],
                                     round(meta['training_seconds'], 2), meta['parameters']]})
    if meta['task'] == 'task3':
        extra = pd.DataFrame({'Thông số': ['Position embedding', 'Flat context', 'Augmentation', 'Crop'],
                              'Giá trị': [config['position'], config['flat_context'], config['augment'], config['crop_mode']]})
        table = pd.concat([table, extra], ignore_index=True)
        if config.get('architecture') in ['aligned','refined']:
            extra = pd.DataFrame({'Thông số': ['Architecture', 'Local-region mixture', 'CNN learning rate', 'Height-projection learning rate'],
                                  'Giá trị': [config['architecture']+' CNN–GRU', config['alignment'], config['cnn_learning_rate'], config['projection_learning_rate']]})
            table = pd.concat([table, extra], ignore_index=True)
    return table


def validation_comparison(task):
    if task == 'task3':
        table = pd.DataFrame(plate_runs()).drop(columns='path')
        table['Trạng thái'] = 'Validation hoàn tất'
        partial = []
        for path in sorted((ROOT/'run/task3').glob('*/*/*/metadata.json')):
            meta = read_json(path)
            if not meta.get('validation_evaluated'):
                config = read_json(path.parent/'config.json')
                partial.append({'model': config['name'], 'seed': config['seed'],
                                'plate_accuracy': np.nan, 'CER': np.nan,
                                'Trạng thái': 'Chưa hoàn tất validation'})
        if partial:
            table = pd.concat([table, pd.DataFrame(partial)], ignore_index=True)
        table['Full-plate (%)'] = table.pop('plate_accuracy')*100
        table['CER (%)'] = table.pop('CER')*100
        table = table.rename(columns={'model': 'Cấu hình'}).round(3)
        return table.fillna('—')
    table = pd.read_csv(ROOT/'run/validation_comparisons.csv')
    return table[table.task.eq(task)][['name', 'validation_score', 'selected']].rename(
        columns={'name': 'Cấu hình', 'validation_score': 'Validation RMSE', 'selected': 'Được chọn'}).round(4)


def forecast_benchmark(task):
    name = read_json(ROOT/'run/selected_configs.json')[task]['config']['name']
    table = pd.read_csv(ROOT/'run/summary/all_test_metrics.csv')
    return table[table.task.eq(task) & table.name.eq(name)].copy()


def plate_test_comparison():
    selection=read_json(ROOT/'run/task3/final_selection.json')
    rows=[]
    for item in selection['checkpoints']:
        path=ROOT/item['run']
        config=read_json(path/'config.json')
        scores=read_json(path/'test_metrics.json')
        meta=read_json(path/'metadata.json')
        rows.append({'Model':config['name'],'Seed':config['seed'],
                     'Full-plate (%)':100*scores['plate_accuracy'],
                     'Character (%)':100*scores['character_accuracy'],
                     'CER (%)':100*scores['CER'],'Parameters':meta['parameters']})
    return pd.DataFrame(rows).round(3)


def benchmark_tables(table):
    metrics = ['R2', 'NSE', 'MAE', 'RMSE']
    macro = table[table.horizon.eq('macro')]
    seeds = macro[macro.model.eq('model')][['seed']+metrics].set_index('seed')
    rows = []
    for model, label in [('persistence', 'Persistence'), ('daily_seasonal', 'Daily seasonal')]:
        base = macro[macro.model.eq(model)].iloc[0]
        rows.append({'Mô hình': label, **{m: f'{base[m]:.4f}' for m in metrics}})
    rows.append({'Mô hình': 'Model — mean ± std (3 seed)',
                 **{m: f'{seeds[m].mean():.4f} ± {seeds[m].std(ddof=1):.4f}' for m in metrics}})
    persistence = macro[macro.model.eq('persistence')].iloc[0].RMSE
    improvement = 100*(1-seeds.RMSE.mean()/persistence)
    return pd.DataFrame(rows), seeds.round(4), improvement


def plate_examples(data, predictions=None):
    import matplotlib.pyplot as plt
    from PIL import Image
    if predictions is None:
        rows = data[data.split.eq('train')].head(6).copy()
    else:
        chosen = pd.concat([predictions[predictions.exact_match].head(3),
                            predictions[~predictions.exact_match].head(3)])
        rows = chosen.merge(data[['image_path', 'bbox']], on='image_path', validate='one_to_one')
    fig, axes = plt.subplots(2, 3, figsize=(11, 4))
    for ax in axes.flat:
        ax.axis('off')
    for ax, (_, row) in zip(axes.flat, rows.iterrows()):
        with Image.open(ROOT/row.image_path) as source:
            ax.imshow(source.crop(json.loads(row.bbox)))
        title = row['text'] if predictions is None else row.true_text+' → '+row.predicted_text
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    plt.show()


def plate_batch(data, vocabulary, count=2):
    from PIL import Image
    from utils import ROOT
    images, labels = [], []
    lookup = {char: index for index, char in enumerate(vocabulary)}
    for _, row in data[data.split.eq('train')].head(count).iterrows():
        with Image.open(ROOT/row.image_path) as source:
            crop = source.convert('RGB').crop(json.loads(row.bbox))
            crop = crop.resize((192, 64), Image.Resampling.BILINEAR)
            images.append(np.asarray(crop).copy())
        labels.append([lookup[c] for c in row.text])
    X = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float()/255
    return X, torch.tensor(labels)


def prepare_training(model, seed):
    from utils import windows_device
    return model.to(windows_device(seed))


def train_presented_model(task, model, optimizer, settings, parts=None, scalers=None):
    # Same fit routines as the paired scripts. No test evaluation in a new trial.
    from utils import (safe_training_config, new_run, save_json, prepare_forecast,
                       fit_forecast, evaluate_forecast, prepare_plate_cache, PlateDataset,
                       plate_loader, fit_plates, evaluate_plates)
    config = safe_training_config({**settings, 'name': settings['name']+'_notebook_trial', 'epochs': 2,
                                  'initial_checkpoint':None,'initialization':'new_from_scratch_notebook_trial'})
    device = next(model.parameters()).device
    if device.type != 'cuda':
        raise RuntimeError('Training requires the monitored launcher.')
    if task != 'task3':
        _, _, horizons, identity = prepare_forecast(task, config)
        path = new_run(task, config, identity)
        save_json(path/'scalers.json', scalers)
        fit_forecast(model, optimizer, parts, config, scalers, path, device)
        evaluate_forecast(model, parts, scalers, horizons, path, device, 'validation')
    else:
        frame, vocab, cache, identity = prepare_plate_cache(config['crop_mode'])
        loaders = {s: plate_loader(PlateDataset(frame, vocab, cache, s,
                   augment=config['augment'] and s=='train'), config, shuffle=s=='train')
                   for s in ['train', 'validation']}
        path = new_run(task, config, identity+':'+config['crop_mode'])
        save_json(path/'vocabulary.json', vocab)
        fit_plates(model, optimizer, loaders, config, vocab, path, device)
        evaluate_plates(model, loaders['validation'], frame, vocab, path, device, 'validation')
    return load_run(task, path)
