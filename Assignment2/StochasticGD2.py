import numpy as np
import matplotlib.pyplot as plt
from PreprocessData import load37
from Equations import *
import time

# avoid overflow warnings
np.seterr(all="ignore")

# load train and test data:
x_training, t_training = load37(version="train")
x_test, t_test = load37(version="test")

lb = 80
ub = 100
x = x_training[:lb]
t = t_training[:lb]
x_val = x_training[lb+1:ub]
t_val = t_training[lb+1:ub]

# store dimensions of data:
N = np.shape(x)[0]
d = np.shape(x)[1]

# set parameters:
decay = 0
epochs = 10000
eta = 0.1
alpha = 0.9
batch_size = 0.01 * ub

# initialize weights and gradients
w, wm = (np.random.randn(1, np.shape(x)[1]) for weights in range(2))
dW, dWm = (np.random.randn(1) for i in range(2))

# initialize predictions:
y, ym = (np.random.randn(1,N) for i in range(2))
y_val, ym_val = (np.random.randn(1,ub-lb-1) for i in range(2))

# Start time:
start = time.time()

# initialize loss arrays:
train_loss, train_loss_m = (np.zeros(epochs) for i in range(2))
val_loss, val_loss_m = (np.zeros(epochs) for i in range(2))
xaxis = []

counter = 0
# start training/validation:
for epoch in range(epochs):
    # get minibatches:
    begin = int((epoch*batch_size)%lb)
    end = int(begin + batch_size)
    x_batch = x[begin:end]
    t_batch = t[begin:end]

    begin_val = int((epoch*batch_size)%(ub-(lb+1)))
    end_val = int(begin_val + batch_size)
    x_batch_val = x_val[begin_val:end_val]
    t_batch_val = t_val[begin_val:end_val]

    # forward computation:
    y[0, begin:end] = forward(np.transpose(x_batch), w)
    y_val[0, begin_val:end_val] = forward(np.transpose(x_batch_val), w)
    ym[0, begin:end] = forward(np.transpose(x_batch), wm)
    ym_val[0, begin_val:end_val] = forward(np.transpose(x_batch_val), wm)

    # backward propagation:
    gradE = backward(x_batch, y[0, begin:end], t_batch)
    dW = -eta * gradE
    w = w + dW

    gradE_m = backward(x_batch, ym[0, begin:end], t_batch)
    dWm = -eta * gradE + alpha * dWm
    wm = wm + dWm

    train_loss[epoch] = cost(y, t)
    val_loss[epoch] = cost(y_val, t_val)
    train_loss_m[epoch] = cost(ym, t)
    val_loss_m[epoch] = cost(ym_val, t_val)
    xaxis.append(epoch)

# stop time:
end = time.time()
print "time: ", end - start

plt.figure()
plt.subplot(1, 2, 1)
ytest = forward(np.transpose(x_test), w)
tcost = cost(ytest, t_test)
plt.plot(xaxis, train_loss, c='royalblue', label='train')
plt.plot(xaxis, val_loss, c='darkorange', label='validation')
plt.plot(xaxis, [tcost] * epochs, c='grey', linestyle='--', label='test')
plt.title("Stochastic gradient descent (eta=%1.2f)" %eta)
plt.xlabel("N")
plt.ylabel("loss")

plt.subplot(1, 2, 2)
ytest_m = forward(np.transpose(x_test), wm)
tcost_m = cost(ytest_m, t_test)
plt.plot(xaxis, train_loss_m, c='royalblue', label='train')
plt.plot(xaxis, val_loss_m, c='darkorange', label='validation')
plt.plot(xaxis, [tcost_m] * epochs, c='grey', linestyle='--', label='test')
plt.title("Momentum (alpha=%1.2f)" %alpha)
plt.xlabel("N")
plt.ylabel("loss")
plt.show()
