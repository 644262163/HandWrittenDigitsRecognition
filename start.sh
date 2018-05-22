!/bin/bash

echo "训练模型开始"
python train.py
echo "训练模型结束"

echo "测试模型开始"
python test.py
echo "测试模型结束"

echo "图表生成开始"
python board.py
echo "图表生成结束"
