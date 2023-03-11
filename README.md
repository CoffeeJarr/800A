### 2023/03/11

- [ ]  Pytorch学习：图像卷积、前向后向、层与块等
- [ ]  Jester 数据集完成处理
- [ ]  远程配置NVIDIA GPU 4080进行预训练测试
- [ ]  简单 RGB 模式完成多种数量 sample segment 训练
    
    ```
    CUDA_VISIBLE_DEVICES=0,1 python main.py jester RGB \
                         --arch BNInception --num_segments 2 \
                         --consensus_type TRN --batch-size 64
    CUDA_VISIBLE_DEVICES=0,1 python main.py jester RGB \
                         --arch BNInception --num_segments 3 \
                         --consensus_type TRN --batch-size 64
    CUDA_VISIBLE_DEVICES=0,1 python main.py jester RGB \
                         --arch BNInception --num_segments 4 \
                         --consensus_type TRN --batch-size 64
    CUDA_VISIBLE_DEVICES=0,1 python main.py jester RGB \
                         --arch BNInception --num_segments 5 \
                         --consensus_type TRN --batch-size 64
    ```
    

### Next steps:

- [ ]  对训练结果进行分析，采用合适的非线性方法进行修改
- [ ]  接上替换BNInception为VGG，进行训练
- [ ]  上述并行：中期报告撰写
