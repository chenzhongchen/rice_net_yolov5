import os

label_folder = '../yolov5-5.0/rice_data/labels/train/'  # 替换为你的标签路径
image_folder = '../yolov5-5.0/rice_data/images/train/'  # 替换为你的图像路径

# 检查每个图像是否都有对应的标签
for img_name in os.listdir(image_folder):
    if img_name.endswith('.jpg'):  # 假设图像为 jpg 格式
        label_name = img_name.replace('.jpg', '.txt')
        label_path = os.path.join(label_folder, label_name)

        if not os.path.exists(label_path):
            print(f'Missing label for image: {img_name}')
        else:
            # 检查标签文件内容并打印
            with open(label_path, 'r') as f:
                lines = f.readlines()
                if not lines:
                    print(f'Label file is empty: {label_name}')
                else:
                    print(f'Contents of {label_name}:')
                    for line in lines:
                        print(line.strip())