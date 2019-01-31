import matplotlib.pyplot as plt
from PreprocessData import load37
from Equations import *


# avoid overflow warnings
np.seterr(all="ignore")

# load train (N=12396L) and test (N=2038) data
x_training, t_training = load37(version="train")
x_test, t_test = load37(version="test")

# split 7:1 ratio
lb = np.round(np.shape(x_training)[0] * 7 / 8)
ub = np.shape(x_training)[0]
x = x_training[:lb]
t = t_training[:lb]
x_val = x_training[lb+1:ub]
t_val = t_training[lb+1:ub]

# parameters
eta = 1
alpha = 0.8

# momentum, stats: train - val - test
tloss_m, ytrain_m, vloss_m, yval_m, weight_vector_m, end_m, start_m, n_epochs_m = training(x, t, x_val, t_val, 1, 0, eta, alpha=alpha)
ytest_m = forward(np.transpose(x_test), weight_vector_m)
tcost_m = cost(ytest_m, t_test)
print "Loss: ", tloss_m[n_epochs_m-1], vloss_m[n_epochs_m-1], tcost_m
print "Error: ", classification_error(ytrain_m, t), classification_error(yval_m, t_val), classification_error(ytest_m, t_test)
print "Acc: ", testing(x, weight_vector_m, t), "%", testing(x_val, weight_vector_m, t_val), "%", testing(x_test, weight_vector_m, t_test), "%"
print "Time: ", end_m - start_m
print "itr: ", n_epochs_m

# gradient descent with momentum
plt.figure()
xaxis = range(0, n_epochs_m)
plt.plot(xaxis, tloss_m, c='royalblue')
plt.plot(xaxis, vloss_m, c='darkorange')
plt.plot(xaxis, [tcost_m] * n_epochs_m, c='grey', linestyle='--')
plt.legend(["train", "validation", "test"])
plt.title("Momentum (alpha=%1.2f)" % alpha)
plt.xlabel("N")
plt.ylabel("loss")
plt.show()



