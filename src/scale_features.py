from utils import *

X_train = pd.read_csv('results/X_train.csv')
X_test = pd.read_csv('results/X_test.csv')

mean = X_train.mean()
std = X_train.std()

# scale
X_train_scaled = (X_train - mean) / std
X_test_scaled = (X_test - mean) / std

# save to csv
X_train_scaled.to_csv('results/X_train_scaled.csv', index=False)
X_test_scaled.to_csv('results/X_test_scaled.csv', index=False)

# save mean & std for inference
mean.to_csv('results/X_mean.csv', index=True)
std.to_csv('results/X_std.csv', index=True)