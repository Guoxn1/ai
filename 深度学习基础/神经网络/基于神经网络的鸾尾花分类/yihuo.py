import numpy as np

# 定义激活函数（Sigmoid）
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 输入数据
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# 目标输出
y = np.array([[0], [1], [1], [0]])

# 设置随机种子，以确保结果可重复
np.random.seed(42)

# 初始化权重和偏置
W1 = np.random.random((2, 2))
b1 = np.zeros((1, 2))
W2 = np.random.random((2, 1))
b2 = np.zeros((1, 1))

# 定义学习率和迭代次数
learning_rate = 0.1
epochs = 10000

# 训练网络
for epoch in range(epochs):
    # 前向传播
    hidden_layer_input = np.dot(X, W1) + b1
    hidden_layer_output = sigmoid(hidden_layer_input)
    # hidden_layer_output = hidden_layer_input
    output_layer_input = np.dot(hidden_layer_output, W2) + b2
    # output_layer_output = output_layer_input
    output_layer_output = sigmoid(output_layer_input)

    # 计算损失值
    # loss = np.mean((output_layer_output - y) ** 2)

    # 反向传播
    # sigmoid_derivative(x) = sigmoid(x) * (1 - sigmoid(x))
    # 根据损失函数计算此处的梯度
    output_error = (output_layer_output - y) * output_layer_output * (1 - output_layer_output)
    # 链式法则计算隐藏层的梯度
    hidden_layer_error = np.dot(output_error, W2.T) * hidden_layer_output * (1 - hidden_layer_output)

    # 更新权重和偏置
    # 权重矩阵 W2 减去学习率乘以隐藏层输出的转置与输出层误差项的乘积，即更新权重矩阵 W2
    W2 -= learning_rate * np.dot(hidden_layer_output.T, output_error)
    b2 -= learning_rate * np.sum(output_error, axis=0, keepdims=True)
    W1 -= learning_rate * np.dot(X.T, hidden_layer_error)
    b1 -= learning_rate * np.sum(hidden_layer_error, axis=0, keepdims=True)

# 用训练好的网络进行预测
hidden_layer_input = np.dot(X, W1) + b1
hidden_layer_output = sigmoid(hidden_layer_input)
output_layer_input = np.dot(hidden_layer_output, W2) + b2
output_layer_output = sigmoid(output_layer_input)

# 输出每个输入对应的计算数值
for i in range(len(X)):
    print("输入:", X[i])
    print("hidden_layer_input", hidden_layer_input[i])
    print("hidden_layer_output", hidden_layer_output[i])
    print("output_layer_input", output_layer_input[i])
    print("output_layer_output", output_layer_output[i])
    print("输出:", output_layer_output[i][0])
    print()

print('W1-----')
print(W1)
print('b1-----')
print(b1)

print('W2-----')
print(W2)
print('b2-----')
print(b2)