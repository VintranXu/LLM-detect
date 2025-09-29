import argparse
import sys
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.nn import functional as F

from tqdm import tqdm
import numpy as np
import os
import time
import random
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
# import seaborn as sns

from models import Autoformer, DLinear, TimeLLM
from data_provider.data_factory import data_provider
from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

parser = argparse.ArgumentParser(description='Time-LLM Classification')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# basic config
parser.add_argument('--task_name', type=str, required=True, default='classification',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model_comment', type=str, required=True, default='TimeLLM-Weather-Local', help='prefix when saving test results')
parser.add_argument('--model', type=str, required=True, default='TimeLLM',
                    help='model name, options: [Autoformer, DLinear, TimeLLM]')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# data loader
parser.add_argument('--data', type=str, required=True, default='UEA', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='Heartbeat', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; '
                         'M:multivariate predict multivariate, S: univariate predict univariate, '
                         'MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--loader', type=str, default='UEA', help='dataset type')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--downsample_step', type=int, default=4)

# forecasting task (keep for compatibility)
parser.add_argument('--pred_len', type=int, default=0, help='prediction sequence length')
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# classification task
parser.add_argument('--num_classes', type=int, default=15, help='number of classes')
parser.add_argument('--is_classification', type=int, default=1, help='is classification task')
parser.add_argument('--custom_prototypes', type=str, nargs='*', default=[], 
                    help='custom prototype words for domain-specific knowledge')

# model define
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--prompt_domain', type=int, default=1, help='')
parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model') # LLAMA, GPT2, BERT
parser.add_argument('--llm_dim', type=int, default='4096', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='CrossEntropy', help='loss function for classification')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--percent', type=int, default=100)

# Classification specific arguments
parser.add_argument('--class_weights', action='store_true', help='use class weights for imbalanced datasets')
parser.add_argument('--label_smoothing', type=float, default=0.0, help='label smoothing for cross entropy')
parser.add_argument('--focal_loss', action='store_true', help='use focal loss instead of cross entropy')   # 默认值是False
parser.add_argument('--focal_alpha', type=float, default=1.0, help='focal loss alpha parameter')
parser.add_argument('--focal_gamma', type=float, default=2.0, help='focal loss gamma parameter')

args = parser.parse_args()

# Validate classification arguments
if args.task_name == 'classification':
    assert args.is_classification == 1, "is_classification must be 1 for classification task"
    assert args.num_classes > 1, "num_classes must be greater than 1"
    assert args.pred_len == 0, "pred_len should be 0 for classification task"

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
def prepare_labels(batch_y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Normalize labels to 0..num_classes-1 and validate range.
    - If all labels are in [1, num_classes], shift to [0, num_classes-1].
    - Ensure shape is (B,).
    """
    batch_y = batch_y.long()
    if batch_y.ndim > 1:
        batch_y = batch_y.squeeze(-1)
    if batch_y.ndim == 0:
        batch_y = batch_y.unsqueeze(0)

    # If labels are strictly 1..K, convert to 0..K-1
    if batch_y.numel() > 0:
        min_y = torch.min(batch_y)
        max_y = torch.max(batch_y)
        if min_y >= 1 and max_y <= num_classes:
            batch_y = batch_y - 1

        # Validate range
        if torch.any(batch_y < 0) or torch.any(batch_y >= num_classes):
            unique_vals = torch.unique(batch_y).tolist()
            raise ValueError(f"Labels out of range 0..{num_classes-1}. Got unique labels: {unique_vals}")
    return batch_y



def compute_classification_metrics(y_true, y_pred, num_classes):
    """Compute comprehensive classification metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'macro_precision': precision,
        'macro_recall': recall,
        'macro_f1': f1,
        'micro_precision': precision_micro,
        'micro_recall': recall_micro,
        'micro_f1': f1_micro
    }


# def plot_confusion_matrix(y_true, y_pred, num_classes, save_path):
#     """Plot and save confusion matrix"""
#     cm = confusion_matrix(y_true, y_pred)
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#     plt.title('Confusion Matrix')
#     plt.ylabel('True Label')
#     plt.xlabel('Predicted Label')
#     plt.savefig(save_path)
#     plt.close()


def vali_classification(args, accelerator, model, vali_data, vali_loader, criterion):
    """Validation function specifically for classification"""
    total_loss = []
    all_predictions = []
    all_targets = []
    
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            desired_dtype = torch.bfloat16 if accelerator.mixed_precision == 'bf16' else torch.float32
            batch_x = batch_x.to(accelerator.device, dtype=desired_dtype)
            batch_y = prepare_labels(batch_y, args.num_classes).to(accelerator.device)
            batch_x_mark = batch_x_mark.to(accelerator.device, dtype=desired_dtype)
            batch_y_mark = batch_y_mark.to(accelerator.device, dtype=desired_dtype)

            # Model prediction
            outputs = model(batch_x, batch_x_mark, None, None)
            
            # Compute loss
            loss = criterion(outputs, batch_y)
            total_loss.append(loss.item())
            
            # Store predictions and targets
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    # Compute metrics
    metrics = compute_classification_metrics(all_targets, all_predictions, args.num_classes)
    
    return np.average(total_loss), metrics


def get_class_weights(train_loader, num_classes, device):
    """Compute class weights for imbalanced datasets"""
    class_counts = torch.zeros(num_classes)
    total_samples = 0
    
    for _, (_, batch_y, _, _) in enumerate(train_loader):
        batch_y = prepare_labels(batch_y, num_classes)
        for class_idx in range(num_classes):
            class_counts[class_idx] += (batch_y == class_idx).sum().item()
        total_samples += batch_y.size(0)
    
    # Compute inverse frequency weights
    class_weights = total_samples / (num_classes * class_counts)
    class_weights = class_weights / class_weights.sum() * num_classes  # Normalize
    
    return class_weights.to(device)


for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.llm_layers,
        args.patch_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.des, ii)

    # Load data
    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')

    path = os.path.join(args.checkpoints, setting + '-' + args.model_comment)
    args.content = load_content(args)
    if not os.path.exists(path) and accelerator.is_local_main_process:
        os.makedirs(path)

    # Initialize model
    if args.model == 'Autoformer':
        model = Autoformer.Model(args).float()
    elif args.model == 'DLinear':
        model = DLinear.Model(args).float()
    else:
        model = TimeLLM.Model(args, custom_prototypes=args.custom_prototypes).float()



    time_now = time.time()

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

    # Get trainable parameters
    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)

    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

    # Learning rate scheduler
    if args.lradj == 'COS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
    else:
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=args.pct_start,
                                            epochs=args.train_epochs,
                                            max_lr=args.learning_rate)

    # Classification-specific loss function
    if args.focal_loss:
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    else:
        if args.class_weights:
            # Compute class weights
            class_weights = get_class_weights(train_loader, args.num_classes, accelerator.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Prepare for distributed training
    train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
        train_loader, vali_loader, test_loader, model, model_optim, scheduler)

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    # Training loop
    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []
        all_train_predictions = []
        all_train_targets = []

        model.train()
        epoch_time = time.time()
        
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
            iter_count += 1
            model_optim.zero_grad()

            # 保持输入与混合精度设置一致（bf16 或 fp32）
            desired_dtype = torch.bfloat16 if accelerator.mixed_precision == 'bf16' else torch.float32
            batch_x = batch_x.to(accelerator.device, dtype=desired_dtype)
            batch_y = prepare_labels(batch_y, args.num_classes).to(accelerator.device)
            batch_x_mark = batch_x_mark.to(accelerator.device, dtype=desired_dtype)
            batch_y_mark = batch_y_mark.to(accelerator.device, dtype=desired_dtype)

            # Forward pass
            if args.use_amp:
                with accelerator.autocast():
                    outputs = model(batch_x, batch_x_mark, None, None)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
            else:
                outputs = model(batch_x, batch_x_mark, None, None)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

            # Store predictions for training metrics
            with torch.no_grad():
                predictions = torch.argmax(outputs, dim=1)
                all_train_predictions.extend(predictions.cpu().numpy())
                all_train_targets.extend(batch_y.cpu().numpy())

            if (i + 1) % 100 == 0:
                accelerator.print(
                    "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            # Backward pass
            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()
            else:
                accelerator.backward(loss)
                model_optim.step()

            if args.lradj == 'TST':
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                scheduler.step()

        accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        
        # Compute training metrics
        train_loss = np.average(train_loss)
        train_metrics = compute_classification_metrics(all_train_targets, all_train_predictions, args.num_classes)
        
        # Validation
        vali_loss, vali_metrics = vali_classification(args, accelerator, model, vali_data, vali_loader, criterion)
        test_loss, test_metrics = vali_classification(args, accelerator, model, test_data, test_loader, criterion)
        
        accelerator.print(
            "Epoch: {0} | Train Loss: {1:.7f} Train Acc: {2:.4f} | "
            "Vali Loss: {3:.7f} Vali Acc: {4:.4f} | Test Loss: {5:.7f} Test Acc: {6:.4f}".format(
                epoch + 1, train_loss, train_metrics['accuracy'], 
                vali_loss, vali_metrics['accuracy'], test_loss, test_metrics['accuracy']))

        # Early stopping based on validation loss
        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            accelerator.print("Early stopping")
            break

        # Learning rate adjustment
        if args.lradj != 'TST':
            if args.lradj == 'COS':
                scheduler.step()
                accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                if epoch == 0:
                    args.learning_rate = model_optim.param_groups[0]['lr']
                    accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)
        else:
            accelerator.print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

    # Final evaluation and save results
    if accelerator.is_local_main_process:
        # 清理GPU内存
        torch.cuda.empty_cache()

        # Load best model
        best_model_path = path + '/' + 'checkpoint.pth'
        # model.load_state_dict(torch.load(best_model_path))

        # 使用 map_location 将模型加载到当前进程的正确设备上
        state_dict = torch.load(best_model_path, map_location=accelerator.device)
        # 在加载 state_dict 之前，最好先获取 unwrap 后的模型
        model = accelerator.unwrap_model(model)
        model.load_state_dict(state_dict)
        
        # Final test evaluation
        test_loss, test_metrics = vali_classification(args, accelerator, model, test_data, test_loader, criterion)
        
        # Get predictions for confusion matrix
        model.eval()
        all_predictions = []
        all_targets = []
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                desired_dtype = torch.bfloat16 if accelerator.mixed_precision == 'bf16' else torch.float32
                batch_x = batch_x.to(accelerator.device, dtype=desired_dtype)
                batch_y = prepare_labels(batch_y, args.num_classes).to(accelerator.device)
                batch_x_mark = batch_x_mark.to(accelerator.device, dtype=desired_dtype)
                batch_y_mark = batch_y_mark.to(accelerator.device, dtype=desired_dtype)

                outputs = model(batch_x, batch_x_mark, None, None)
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
        
        # Print final results
        accelerator.print("="*50)
        accelerator.print("FINAL TEST RESULTS:")
        accelerator.print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        accelerator.print(f"Test Macro F1: {test_metrics['macro_f1']:.4f}")
        accelerator.print(f"Test Micro F1: {test_metrics['micro_f1']:.4f}")
        accelerator.print("="*50)
        
        # Save detailed classification report
        report = classification_report(all_targets, all_predictions)
        with open(os.path.join(path, 'classification_report.txt'), 'w') as f:
            f.write(report)
        
        # # Plot and save confusion matrix
        # cm_path = os.path.join(path, 'confusion_matrix.png')
        # plot_confusion_matrix(all_targets, all_predictions, args.num_classes, cm_path)
        
        # Save final model
        save_dir = './saved_models'
        os.makedirs(save_dir, exist_ok=True)
        final_model_path = os.path.join(save_dir, f'{setting}_final.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'args': args,
            'setting': setting,
            'test_metrics': test_metrics,
            'best_vali_loss': early_stopping.val_loss_min
        }, final_model_path)
        
        accelerator.print(f'Model saved to {final_model_path}')

        # Analyze prototype usage if custom prototypes are used
        if args.custom_prototypes and hasattr(model, 'get_prototype_analysis'):
            # Get a sample batch for analysis
            sample_batch = next(iter(test_loader))
            desired_dtype = torch.bfloat16 if accelerator.mixed_precision == 'bf16' else torch.float32
            sample_x = sample_batch[0][:1].to(accelerator.device, dtype=desired_dtype)  # Take first sample
            
            prototype_analysis = model.get_prototype_analysis(sample_x)
            if prototype_analysis is not None:
                accelerator.print("\nPrototype Usage Analysis:")
                for i, (name, attention) in enumerate(zip(prototype_analysis['prototype_names'], 
                                                         prototype_analysis['average_attention'])):
                    accelerator.print(f"Prototype '{name}': {attention:.4f}")

# Wait for everyone and cleanup
accelerator.wait_for_everyone()
if accelerator.is_local_main_process:
    path = './checkpoints'
    del_files(path)
    accelerator.print('Success delete checkpoints')