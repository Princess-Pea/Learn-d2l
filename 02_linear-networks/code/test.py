import os
import torch
import torchvision

root = "/home/syalis/d2l_Code/d2l-pytorch/pytorch/data"
expected_files = [
    'train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz',
    't10k-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz'
]

print("--- 开始深度诊断 ---")
raw_folder = os.path.join(root, 'FashionMNIST', 'raw')
print(f"1. 目标文件夹路径: {raw_folder}")

if not os.path.exists(raw_folder):
    print("错误: 文件夹不存在！")
else:
    print("2. 文件夹内实际存在的文件清单:")
    actual_files = os.listdir(raw_folder)
    for f in actual_files:
        f_path = os.path.join(raw_folder, f)
        print(f"   - {f} (大小: {os.path.getsize(f_path)} bytes)")

    print("\n3. 检查 PyTorch 预期的文件是否存在:")
    for ef in expected_files:
        exists = ef in actual_files
        print(f"   是否找到 {ef} ? {'[ OK ]' if exists else '[ 缺失 ]'}")

print("\n4. 尝试通过 torchvision 强制加载:")
try:
    # 注意：这里我们主动跳过 download，只看能不能识别本地
    ds = torchvision.datasets.FashionMNIST(root=root, train=True, download=False)
    print("恭喜！识别成功。")
except Exception as e:
    print(f"识别失败，内部报错: {e}")