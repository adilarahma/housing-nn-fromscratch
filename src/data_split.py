from utils import *

df = pd.read_csv('results/housing_clean.csv')

X = df.drop(columns=['price'])
y = df['price']

# trains-test split 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# save to results
X_train.to_csv('results/X_train.csv', index=False)
X_test.to_csv('results/X_test.csv', index=False)
y_train.to_csv('results/y_train.csv', index=False)
y_test.to_csv('results/y_test.csv', index=False)

print("=== Train/test split done ===")
print("X_train: ", X_train.shape, "\tX_test: ", X_test.shape)
print("y_train: ", y_train.shape, "\ty_test: ", y_test.shape)