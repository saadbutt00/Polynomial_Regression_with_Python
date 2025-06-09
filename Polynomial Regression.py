import numpy as np

num_features = int(input('Enter Number of Features - '))
degree = int(input('Enter Polynomial Degree - '))
bias = float(input('Enter Bias value - '))

X = []
for i in range(num_features):
    x = list(map(float, input(f'Enter X{i} Feature values (space-separated): ').split()))
    X.append(x)

X = np.transpose(X)

poly_features = [np.full(X.shape[0], bias)]

for i in range(1, degree + 1):
    for j in range(num_features):
        poly_features.append(X[:, j] ** i)

X_poly = np.transpose(np.vstack(poly_features))
Y = np.array(list(map(float, input('Enter Y values (space-separated): ').split())))

if len(Y) != len(X_poly):
    print('Error - Number of values in each feature should be equal')
    exit()

XT = np.transpose(X_poly)
part1 = np.linalg.pinv(np.dot(XT, X_poly))
part2 = np.dot(XT, Y)
theta = np.dot(part1, part2)

y_pred = np.dot(X_poly, theta)

print("Theta (coefficients):", np.round(theta, 4))
print("Predictions:", np.round(y_pred, 4))

r2_1 = np.sum((Y - y_pred) ** 2)
r2_2 = np.sum((Y - np.mean(Y))** 2)

r2 = 1 - (r2_1/r2_2 + 1e-10)
print('R2 Score - ',np.round(r2*100, 2),'%')
