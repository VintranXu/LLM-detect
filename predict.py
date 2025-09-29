import argparse
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from models import Autoformer, DLinear, TimeLLM
from data_provider.data_factory import data_provider
from utils.tools import StandardScaler, load_content

parser = argparse.ArgumentParser(description='TimeLLM模型预测脚本')

# 基本配置
parser.add_argument('--model_path', type=str, required=True, help='训练好的模型检查点路径')
parser.add_argument('--task_name', type=str, default='long_term_forecast', help='任务名称')
parser.add_argument('--model', type=str, default='TimeLLM', help='模型名称')

# 数据加载器
parser.add_argument('--data', type=str, default='Weather', help='数据集类型')
parser.add_argument('--root_path', type=str, default='./dataset/weather/', help='数据文件根路径')
parser.add_argument('--data_path', type=str, default='weather.csv', help='数据文件')
parser.add_argument('--features', type=str, default='M', help='预测任务类型')
parser.add_argument('--target', type=str, default='OT', help='目标特征')
parser.add_argument('--freq', type=str, default='h', help='时间特征编码频率')

# 预测任务
parser.add_argument('--seq_len', type=int, default=512, help='输入序列长度')
parser.add_argument('--label_len', type=int, default=48, help='开始令牌长度')
parser.add_argument('--pred_len', type=int, default=96, help='预测序列长度')

# 模型定义
parser.add_argument('--enc_in', type=int, default=21, help='编码器输入大小')
parser.add_argument('--dec_in', type=int, default=21, help='解码器输入大小')
parser.add_argument('--c_out', type=int, default=21, help='输出大小')
parser.add_argument('--d_model', type=int, default=32, help='模型维度')
parser.add_argument('--d_ff', type=int, default=32, help='FCN维度')
parser.add_argument('--e_layers', type=int, default=2, help='编码器层数')
parser.add_argument('--d_layers', type=int, default=1, help='解码器层数')
parser.add_argument('--factor', type=int, default=3, help='注意力因子')
parser.add_argument('--llm_layers', type=int, default=32, help='LLM层数')
parser.add_argument('--batch_size', type=int, default=1, help='批处理大小')
parser.add_argument('--output_path', type=str, default='./output', help='输出路径')

# 数据加载器额外参数
parser.add_argument('--embed', type=str, default='timeF', help='时间特征编码方式')
parser.add_argument('--percent', type=float, default=1.0, help='数据使用百分比')
parser.add_argument('--num_workers', type=int, default=0, help='数据加载器工作进程数')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='季节性模式')

# TimeLLM模型额外参数
parser.add_argument('--patch_len', type=int, default=16, help='补丁长度')
parser.add_argument('--stride', type=int, default=8, help='步长')
parser.add_argument('--llm_model', type=str, default='BERT', help='LLM模型类型 (BERT, LLAMA, GPT2)')
parser.add_argument('--llm_dim', type=int, default=768, help='LLM模型维度 (BERT-base:768, LLAMA7b:4096, GPT2-small:768)')
parser.add_argument('--n_heads', type=int, default=8, help='注意力头数')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout率')
parser.add_argument('--prompt_domain', type=int, default=0, help='是否使用提示域')
parser.add_argument('--content', type=str, default='', help='提示内容')
parser.add_argument('--output_attention', action='store_true', help='是否输出注意力权重')
parser.add_argument('--activation', type=str, default='gelu', help='激活函数')
parser.add_argument('--moving_avg', type=int, default=25, help='移动平均窗口大小')

args = parser.parse_args()

def main():
    # 创建输出目录
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    # 打印BERT模型配置信息
    print("=" * 60)
    print("BERT模型配置信息:")
    print(f"LLM模型类型: {args.llm_model}")
    print(f"LLM模型维度: {args.llm_dim}")
    print(f"LLM层数: {args.llm_layers}")
    print(f"补丁长度: {args.patch_len}")
    print(f"步长: {args.stride}")
    print(f"注意力头数: {args.n_heads}")
    print(f"Dropout率: {args.dropout}")
    print(f"激活函数: {args.activation}")
    print("=" * 60)
    
    # 加载测试数据
    _, test_loader = data_provider(args, 'test')
    
    # 根据模型类型初始化模型
    if args.model == 'Autoformer':
        model = Autoformer.Model(args).float()
    elif args.model == 'DLinear':
        model = DLinear.Model(args).float()
    else:
        model = TimeLLM.Model(args).float()
    
    # 加载训练好的模型参数
    checkpoint = torch.load(args.model_path, map_location='cpu')
    
    # 检查是否是新的保存格式（包含model_state_dict的字典）
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], False)
        # 如果保存的检查点包含args，则更新当前args
        if 'args' in checkpoint:
            saved_args = checkpoint['args']
            for key, value in vars(saved_args).items():
                if not hasattr(args, key):
                    setattr(args, key, value)
        print(f"从检查点加载模型，验证损失: {checkpoint.get('best_vali_loss', 'N/A')}")
    else:
        # 兼容旧的保存格式
        model.load_state_dict(checkpoint, False)
    
    # 加载内容（如果需要）
    if hasattr(args, 'content') and args.content == '':
        args.content = load_content(args)
    
    # 如果有GPU则使用
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 设置为评估模式
    model.eval()
    print(f"模型已设置为评估模式，使用设备: {device}")
    
    # 打印BERT模型详细信息
    if args.llm_model == 'BERT':
        print(f"✅ 使用BERT模型进行预测")
        print(f"   - 模型维度: {args.llm_dim}")
        print(f"   - 模型层数: {args.llm_layers}")
        print(f"   - 补丁长度: {args.patch_len}")
        print(f"   - 步长: {args.stride}")
        print(f"   - 注意力头数: {args.n_heads}")
        print(f"   - 序列长度: {args.seq_len}")
        print(f"   - 预测长度: {args.pred_len}")
        print(f"   - 特征类型: {args.features}")
    else:
        print(f"✅ 使用{args.llm_model}模型进行预测")
    
    print(f"模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 存储预测和真实值
    predictions = []
    true_values = []
    
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            
            # 解码器输入
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float()
            
            # 模型预测
            if args.output_attention:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            # 处理输出
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]
            
            # 保存预测和真实值
            predictions.append(outputs.cpu().numpy())
            true_values.append(batch_y.cpu().numpy())
            
            print(f"处理批次 {i+1}/{len(test_loader)}")
    
    # 合并所有批次的结果
    predictions = np.concatenate(predictions, axis=0)
    true_values = np.concatenate(true_values, axis=0)
    
    # 计算评估指标
    mae = np.mean(np.abs(predictions - true_values))
    mse = np.mean((predictions - true_values) ** 2)
    rmse = np.sqrt(mse)
    
    print(f"测试集上的评估结果:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # 可视化一些预测结果
    plot_predictions(predictions, true_values, args.output_path)
    
    # 保存预测结果
    np.save(os.path.join(args.output_path, 'predictions.npy'), predictions)
    np.save(os.path.join(args.output_path, 'true_values.npy'), true_values)
    
    # 保存评估指标
    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model_path': args.model_path,
        'model_type': args.model,
        'data': args.data,
        'seq_len': args.seq_len,
        'pred_len': args.pred_len
    }
    
    # 保存为文本文件
    with open(os.path.join(args.output_path, 'metrics.txt'), 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    
    print(f"预测结果已保存到 {args.output_path}")

def plot_predictions(predictions, true_values, output_path):
    """绘制预测结果与真实值的对比图"""
    # 选择第一个样本的第一个特征进行可视化
    sample_idx = 0
    feature_idx = 0
    
    pred = predictions[sample_idx, :, feature_idx]
    true = true_values[sample_idx, :, feature_idx]
    
    plt.figure(figsize=(10, 6))
    plt.plot(true, label='真实值', marker='o')
    plt.plot(pred, label='预测值', marker='*')
    plt.legend()
    plt.title('时间序列预测结果')
    plt.xlabel('时间步')
    plt.ylabel('值')
    plt.grid(True)
    
    # 保存图表
    plt.savefig(os.path.join(output_path, 'prediction_plot.png'))
    plt.close()

if __name__ == "__main__":
    main()
