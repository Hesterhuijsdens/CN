import matplotlib.pyplot as plt
from PreprocessData import load37
from Equations import *

# avoid overflow warnings
np.seterr(all="ignore")

# load train (N=12396L) and test (N=2038) data
x_training, t_training = load37(version="train")
x_test, t_test = load37(version="test")

# 80 train : 20 validation
# lb = 9917
# ub = np.shape(x_training)[0] - 1
lb = 640
ub = 800
x = x_training[:lb]
t = t_training[:lb]
x_val = x_training[lb+1:ub]
t_val = t_training[lb+1:ub]

# parameters
n_epochs = 1000
eta = 0.01
alpha = 0.9
N = np.shape(x)
xaxis = range(0, n_epochs)
decay = 0.1

# initialize weights and gradients
w, wm, wwd, wwdm = (np.random.randn(1, np.shape(x)[1]) for weights in range(4))
dW, dWm, dWwd, dWwdm = (np.random.randn(1) for i in range(4))

# regular gradient descent
tloss, ytrain, vloss, yval, weight_vector, end, start = training(n_epochs, x=x, t=t, x_val=x_val, t_val=t_val,
                                                                 w=w, dw=dW, momentum=0, weightdecay=0)
class_err = classification_error(ytrain, t)
print "GD"
print "Train class error: ", class_err
print "Val class error", classification_error(yval, t_val)
print "Train loss: ", tloss[n_epochs-1]
print "Val loss: ", vloss[n_epochs-1]
ytest = forward(np.transpose(x_test), weight_vector)
tcost = cost(ytest, t_test)
print "Test loss: ", tcost
print "Train accuracy: ", testing(x, weight_vector, t), "%"
print "Test accuracy: ", testing(x_test, weight_vector, t_test), "%"
print "Time: ", end - start
print " "

# momentum
tloss_m, ytrain_m, vloss_m, yval_m, weight_vector_m, end_m, start_m = training(n_epochs, x=x, t=t, x_val=x_val,
                                                                               t_val=t_val, w=wm, dw=dWm, alpha=alpha,
                                                                               momentum=1, weightdecay=0)
class_err_m = classification_error(ytrain_m, t)
print "GD+M"
print "Train class error: ", class_err_m
print "Val class error", classification_error(yval_m, t_val)
print "Train loss: ", tloss_m[n_epochs-1]
print "Val loss: ", vloss_m[n_epochs-1]
ytest_m = forward(np.transpose(x_test), weight_vector_m)
tcost_m = cost(ytest_m, t_test)
print "Test loss: ", tcost_m
print "Train accuracy: ", testing(x, weight_vector_m, t), "%"
print "Test accuracy: ", testing(x_test, weight_vector_m, t_test), "%"
print "Time: ", end_m - start_m
print " "

# gradient descent
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(xaxis, tloss, c='royalblue')
plt.plot(xaxis, vloss, c='darkorange')
plt.plot(xaxis, [tcost] * n_epochs, c='grey', linestyle='--')
plt.legend(["train", "validation", "test"])
plt.title("Gradient descent (eta=%1.2f)" %eta)
plt.xlabel("N")
plt.ylabel("loss")

# gradient descent with momentum
plt.subplot(1, 2, 2)
plt.plot(xaxis, tloss_m, c='royalblue')
plt.plot(xaxis, vloss_m, c='darkorange')
plt.plot(xaxis, [tcost_m] * n_epochs, c='grey', linestyle='--')
plt.legend(["train", "validation", "test"])
plt.title("Momentum (alpha=%1.2f)" %alpha)
plt.xlabel("N")
plt.ylabel("loss")
plt.show()



