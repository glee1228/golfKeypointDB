import numpy as np
from matplotlib import pyplot as plt
import csv
from sklearn import linear_model
np.random.seed(0)


n_samples = 1000
n_outliers = 50


def get_polynomial_samples(n_samples=1000):
    X = np.array(range(1000)) / 100.0
    np.random.shuffle(X)

    coeff = np.random.rand(2,) * 3

    y = coeff[0]*X**2 + coeff[1]*X + 10
    X = X.reshape(-1, 1)
    return coeff, X, y


def add_square_feature(X):
    X = np.concatenate([(X**2).reshape(-1,1), X], axis=1)
    return X

frame_list = []
r_list = []
degree_list = []
with open('polar_3.csv', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        frame_list.append(row[0])
        r_list.append(row[1])
        degree_list.append(row[2])


# coef, X, y = get_polynomial_samples(n_samples)

X_train = np.array(frame_list,dtype=np.float64).reshape(len(frame_list),1)
y_train = np.array(r_list,dtype=np.float64)

up_start = 10
up_end = 426
down_end = 650
finishing_end = 778

# up = X[up_start,up_end]
# down = X[up_end:down_end]
# finish = X[down_end:finishing_end]
print(X_train.shape,y_train.shape)
# shot_sequence = [0,426,650,775]
shot_sequence = [0,100,250,426,550,650]

for i in range(len(shot_sequence)-1):
    X_train_copy = X_train.reshape(len(frame_list))
    index = (X_train_copy<shot_sequence[i+1]) & (X_train_copy>shot_sequence[i])

    # import pdb;pdb.set_trace()
    X = X_train[index]
    y = y_train[index]
    # y = np.array(degree_list,dtype=np.float64)
    # X, y =
    print(X.shape, y.shape)
    print(shot_sequence[i],shot_sequence[i+1])


    # Fit line using all data
    lr = linear_model.LinearRegression()
    lr.fit(add_square_feature(X), y)

    # Robustly fit linear model with RANSAC algorithm
    ransac = linear_model.RANSACRegressor()
    ransac.fit(add_square_feature(X), y)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    # Predict data of estimated models
    line_X = np.arange(X.min(), X.max())[:, np.newaxis]
    line_y = lr.predict(add_square_feature(line_X))
    line_y_ransac = ransac.predict(add_square_feature(line_X))

    #Compare estimated coefficients
    print("Estimated coefficients ( linear regression, RANSAC):")
    print(lr.coef_, ransac.estimator_.coef_)

    lw = 2
    plt.scatter(X[inlier_mask], y[inlier_mask], color='yellowgreen', marker='.',
                label='Inliers')
    plt.scatter(X[outlier_mask], y[outlier_mask], color='gold', marker='.',
                label='Outliers')
    plt.plot(line_X, line_y, color='navy', linewidth=lw, label='Linear regressor')
    plt.plot(line_X, line_y_ransac, color='cornflowerblue', linewidth=lw,
             label='RANSAC regressor')
    plt.legend(loc='lower right')
    plt.xlabel("Input")
    plt.ylabel("Response")
    plt.show()