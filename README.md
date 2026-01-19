整体项目目录结构：
<img width="865" height="38" alt="image" src="https://github.com/user-attachments/assets/8714609b-f998-4708-aa2d-6f9c6ffd7647" />
第一步：获取tif图像的宽W和高H
执行python tifInfo.py
<img width="865" height="428" alt="image" src="https://github.com/user-attachments/assets/519fc0b1-46a4-492b-8b15-4db1875fb38b" />
第二步：获取tif图像的预特征
执行python tifFeature.py –h
<img width="865" height="434" alt="image" src="https://github.com/user-attachments/assets/226273b0-a49c-4089-97e0-750b9d64e20e" />
执行正确显示：
<img width="865" height="239" alt="image" src="https://github.com/user-attachments/assets/65a8a535-199f-4237-8caf-792a7c3a62d7" />
执行全过程，获取tif图像预特征：
<img width="865" height="486" alt="image" src="https://github.com/user-attachments/assets/9df0b8a4-ca0f-4d20-8f7c-91cace5df473" />
第三步：训练：各参数说明请参考train.py程序
<img width="865" height="408" alt="image" src="https://github.com/user-attachments/assets/b4c15968-a752-48ff-8c06-70ba88ee1bf1" />
生成最优模型文件best_pollution_model.pth和col_stats.pkl，后者是它是连接“原始数据”和“神经网络维度”的翻译字典。如果没有这个文件，模型在推理（Inference）时就不知道如何将输入的文字或类别转化成正确维度的向量。
第四步：推理inference.py
<img width="865" height="238" alt="image" src="https://github.com/user-attachments/assets/eb67128c-07f1-42c5-8c8e-333c7389a217" />
最后项目目录：
<img width="865" height="47" alt="image" src="https://github.com/user-attachments/assets/76959e07-c070-474a-9473-6559cdf764d2" />


