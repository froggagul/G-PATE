import numpy as np
import matplotlib.pyplot as plt

vanilla_train = np.load('results/b5b3646e-34bd-4b41-9da9-474ac74e4077_train.npy')
vanilla_eval = np.load('results/b5b3646e-34bd-4b41-9da9-474ac74e4077_eval.npy')
var_threshold_train = np.load('results/c2e6020e-d37e-4a5e-bbad-d7f71553ecbd_train.npy')
var_threshold_eval = np.load('results/c2e6020e-d37e-4a5e-bbad-d7f71553ecbd_eval.npy')

plt.plot(vanilla_train, label='base train')
plt.plot(vanilla_eval, label='base eval')
plt.plot(var_threshold_train, label='variance threshold train')
plt.plot(var_threshold_eval, label='variance threshold eval')

plt.legend()
plt.title('base vs variance threshold accuracy graph')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.xticks()
plt.savefig('results/accuracy.png')

