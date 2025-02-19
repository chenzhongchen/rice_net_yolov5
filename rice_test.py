import torch
import cv2
import numpy as np

# 加载本地训练好的 YOLOv5 模型
model = torch.load('best.pt', map_location='cpu')  # 使用 'cpu' 或 'cuda' 取决于你的设备
model.eval()  # 设置模型为评估模式

# 读取图像
img = cv2.imread('123.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式
img = cv2.resize(img, (640, 640))  # 调整图像大小为 640x640

# 将图像转换为张量
img_tensor = torch.from_numpy(img).float() / 255.0  # 归一化
img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # 改变形状为 (1, 3, 640, 640)

# 推理
with torch.no_grad():
    results = model(img_tensor)

# 处理结果
detections = results[0].cpu().numpy()  # 将结果转换为 NumPy 数组

# 检查 detections 的形状
print(detections.shape)

# 遍历检测结果
for detection in detections:
    if detection.size > 0:  # 确保有检测结果
        for obj in detection:  # 遍历每个检测框
            if obj[4] >= 0.25:  # 置信度阈值
                # 提取坐标、置信度和类
                x1, y1, x2, y2 = map(int, obj[:4])  # 前四个值是坐标
                conf = obj[4]  # 置信度
                cls = int(obj[5])  # 类别

                label = f'Class {cls}: {conf:.2f}'

                # 绘制框和标签
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# 显示结果
cv2.imshow('Detections', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()