import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
import os
import numpy as np

# ================= 配置区域 =================
# 确保这里是你的新文件路径
DATASET_PATH = "/home/shenji/uav_payload_lab/uav_payload_lab/logs/rsl_rl/Encoder_DataCollection/2026-01-09_02-02-10/encoder_dataset.pt"

# 因为你有400万数据，Batch Size 可以开大一点，训练更快
BATCH_SIZE = 4096  
LEARNING_RATE = 1e-3
EPOCHS = 100        # 数据量大，可能不需要太多 Epoch 就能收敛
HIDDEN_DIM = 128
# ===========================================

class CNNEncoder(nn.Module):
    def __init__(self, input_dim=21, history_len=50, output_dim=2):
        super().__init__()
        
        # === 1. 1D CNN 特征提取层 ===
        # 输入形状: (Batch, Channels=21, Length=50)
        # 作用: 提取时序特征 (比如: 抖动频率、动作延迟)
        self.cnn_layers = nn.Sequential(
            # 第一层卷积: 感受野较小，提取局部特征
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64), # 加速收敛
            
            # 第二层卷积
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            
            # 第三层卷积
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            
            nn.Flatten() # 展平: 64通道 * 50长度 = 3200
        )
        
        flatten_dim = 64 * history_len
        
        # === 2. MLP 回归预测层 ===
        self.regressor = nn.Sequential(
            nn.Linear(flatten_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim) # 输出预测值
        )

    def forward(self, x):
        # x 原始形状: (Batch, History=50, Dim=21)
        # Conv1d 需要: (Batch, Dim=21, History=50) -> 需要交换维度
        x = x.permute(0, 2, 1)
        
        features = self.cnn_layers(x)
        output = self.regressor(features)
        return output

def main():
    # 检查是否有 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 加载数据
    print(f"Loading dataset from {DATASET_PATH}...")
    if not os.path.exists(DATASET_PATH):
        print(f"Error: File not found at {DATASET_PATH}")
        return

    data = torch.load(DATASET_PATH)
    # 转为 Float 并移到 CPU (先不进 GPU，只有 Batch 进 GPU，防止爆显存)
    inputs = data["inputs"].float()
    labels = data["labels"].float()
    
    total_samples = len(labels)
    print(f"Dataset Loaded. Total Samples: {total_samples}")
    print(f"Input Shape: {inputs.shape}, Label Shape: {labels.shape}")

    # === [关键] 数据正确性自检 ===
    print("\n[Data Inspection] Checking first 5 labels (Mass, Length):")
    print(labels[:5].numpy())
    print("Label Stats:")
    print(f"  Mass -> Mean: {labels[:,0].mean():.4f}, Std: {labels[:,0].std():.4f}, Min: {labels[:,0].min():.4f}, Max: {labels[:,0].max():.4f}")
    print(f"  Len  -> Mean: {labels[:,1].mean():.4f}, Std: {labels[:,1].std():.4f}, Min: {labels[:,1].min():.4f}, Max: {labels[:,1].max():.4f}")
    
    if labels[:,0].std() < 1e-6:
        print("⚠️ WARNING: Mass 标签似乎没有变化 (Std=0)。请检查采集脚本是否开启了 Domain Randomization!")
    # ============================

    # 2. 划分数据集 (90% 训练, 10% 测试)
    # 400万数据量很大，验证集留 10% 足够了
    dataset = TensorDataset(inputs, labels)
    train_size = int(0.9 * total_samples)
    test_size = total_samples - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])
    
    # num_workers=4 可以加速数据加载，但在某些系统上可能报错，如果报错改为 0
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"\nTrain samples: {len(train_set)}, Test samples: {len(test_set)}")

    # 3. 初始化
    model = CNNEncoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss() 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 4. 训练
    best_loss = float('inf')
    history = {'train': [], 'test': []}

    print("\nStart Training...")
    try:
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            avg_train_loss = running_loss / len(train_loader)
            
            # 验证
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    pred = model(batch_x)
                    loss = criterion(pred, batch_y)
                    test_loss += loss.item()
            
            avg_test_loss = test_loss / len(test_loader)
            
            # 记录历史
            history['train'].append(avg_train_loss)
            history['test'].append(avg_test_loss)
            
            # 调整学习率
            scheduler.step(avg_test_loss)

            print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.6f} | Test Loss: {avg_test_loss:.6f}")

            # 保存最优模型
            if avg_test_loss < best_loss:
                best_loss = avg_test_loss
                torch.save(model.state_dict(), "best_cnn_encoder.pth")
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current progress...")

    print(f"\nTraining Finished. Best Test Loss: {best_loss:.6f}")
    
    # 5. 最终效果展示
    print("\n=== Final Validation Check ===")
    model.load_state_dict(torch.load("best_cnn_encoder.pth"))
    model.eval()
    model.to(device)
    
    # 拿一个 batch 出来看一眼
    x, y = next(iter(test_loader))
    x = x[:10].to(device) # 取前10个
    y = y[:10].cpu().numpy()
    
    with torch.no_grad():
        pred = model(x).cpu().numpy()
        
    print(f"{'True Mass':<10} {'Pred Mass':<10} | {'True Len':<10} {'Pred Len':<10} | {'Error':<10}")
    print("-" * 60)
    for i in range(10):
        # 计算误差
        err = np.mean(np.abs(y[i] - pred[i]))
        print(f"{y[i][0]:<10.4f} {pred[i][0]:<10.4f} | {y[i][1]:<10.4f} {pred[i][1]:<10.4f} | {err:<10.4f}")

    # 画图
    plt.figure(figsize=(10, 5))
    plt.plot(history['train'], label='Train Loss')
    plt.plot(history['test'], label='Test Loss')
    plt.yscale('log') # 使用对数坐标，方便看后期微小的下降
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (Log Scale)')
    plt.legend()
    plt.title('CNN Encoder Training Curve')
    plt.savefig('cnn_training_curve.png')
    print("\nTraining curve saved to cnn_training_curve.png")

if __name__ == "__main__":
    main()