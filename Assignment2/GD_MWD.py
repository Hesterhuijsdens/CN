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
alpha = 0.75
decay = 0.1

# weight decay + momentum, stats: train - val - test
tloss_wdm, ytrain_wdm, vloss_wdm, yval_wdm, weight_vector_wdm, end_wdm, start_wdm, n_epochs_wdm = training(x, t, x_val, t_val, 1, 1, eta, decay=decay, alpha=alpha)
ytest_wdm = forward(np.transpose(x_test), weight_vector_wdm)
# tcost_wdm = cost_decay(ytest_wdm, t_test, decay, weight_vector_wdm)
tcost_wdm = cost(ytest_wdm, t_test)
print "Loss: ", tloss_wdm[n_epochs_wdm-1], vloss_wdm[n_epochs_wdm-1], tcost_wdm
print "Error: ", classification_error(ytrain_wdm, t), classification_error(yval_wdm, t_val), classification_error(ytest_wdm, t_test)
print "Acc: ", testing(x, weight_vector_wdm, t), "%", testing(x_val, weight_vector_wdm, t_val), "%", testing(x_test, weight_vector_wdm, t_test), "%"
print "Time: ", end_wdm - start_wdm
print "Itr: ", n_epochs_wdm

# gradient descent
plt.figure()
xaxis = range(0, n_epochs_wdm)
plt.plot(xaxis, tloss_wdm, c='royalblue')
plt.plot(xaxis, vloss_wdm, c='darkorange')
plt.plot(xaxis, [tcost_wdm] * n_epochs_wdm, c='grey', linestyle='--')
plt.legend(["train", "validation", "test"])
plt.title("Weight decay (decay=%1.2f) and momentum (alpha=%1.2f)" % (decay, alpha))
plt.xlabel("N")
plt.ylabel("loss")
plt.show()



