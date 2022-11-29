

import numpy as np

#激活函数（activation function）
def sigmoid(x):
  # Our activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

class Neuron: #神经元结点
  def __init__(self, weights, bias):
    self.weights = weights
    self.bias = bias

# 把神经元的输入向前传递获得输出的过程称为前馈
  def feedforward(self, inputs):
      #向量点积
    # Weight inputs, add bias, then use the activation function
    total = np.dot(self.weights, inputs) + self.bias
    return sigmoid(total)



# weights = np.array([0, 1]) # w1 = 0, w2 = 1
# bias = 4                   # b = 4
# n = Neuron(weights, bias)

# x = np.array([2, 3])       # x1 = 2, x2 = 3


# # w·x+b =（x1 × w1）+（x2 × w2）+ b = 0×2+1×3+4=7
# # y=f(w⋅X+b)=f(7)=0.999
# print(n.feedforward(x))




class OurNeuralNetwork:
  '''
  A neural network with:
    - 2 inputs
    - a hidden layer with 2 neurons (h1, h2)
    - an output layer with 1 neuron (o1)
  Each neuron has the same weights and bias:
    - w = [0, 1]
    - b = 0
  '''
  def __init__(self):
    weights = np.array([0, 1]) #权重（weight）
    bias = 0        #偏置（bias）

    # The Neuron class here is from the previous section
    self.h1 = Neuron(weights, bias)
    self.h2 = Neuron(weights, bias)
    self.o1 = Neuron(weights, bias)

  def feedforward(self, x):
    out_h1 = self.h1.feedforward(x)
    out_h2 = self.h2.feedforward(x)

    # The inputs for o1 are the outputs from h1 and h2
    out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))

    return out_o1

network = OurNeuralNetwork()
# x = np.array([2, 3])

# print(network.feedforward(x)) # 0.7216325609518421
# h1=h2=f(w⋅x+b)=f((0×2)+(1×3)+0)
# =f(3)
# =0.9526

# o1=f(w⋅[h1,h2]+b)=f((0∗h1)+(1∗h2)+0)
# =f(0.9526)
# =0.7216



# 均方误差就是所有数据方差的平均值，定义为损失函数。
# 预测结果越好，损失就越低，训练神经网络就是将损失最小化。
def mse_loss(y_true, y_pred):
  # y_true and y_pred are numpy arrays of the same length.
  return ((y_true - y_pred) ** 2).mean()

# y_true = np.array([1, 0, 0, 1])
# y_pred = np.array([0, 0, 0, 0])

# print("mse_loss",mse_loss(y_true, y_pred)) # 0.5

# 改变网络的权重和偏置可以影响预测值
# η是一个常数，称为学习率（learning rate），
# 它决定了我们训练网络速率的快慢。将w1减去η·∂L/∂w1，就等到了新的权重w1。

# 当∂L/∂w1是正数时，w1会变小；当∂L/∂w1是负数 时，w1会变大。
# 如果我们用这种方法去逐步改变网络的权重w和偏置b，损失函数会缓慢地降低，从而改进神经网络。
# 训练流程如下：
# 1、从数据集中选择一个样本；
# 2、计算损失函数对所有权重和偏置的偏导数；
# 3、使用更新公式更新每个权重和偏置；
# 4、回到第1步。



