import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

# 生成隨機數據
rng = np.random.RandomState(42)
X = 0.3 * rng.randn(100, 2)
X_train = np.r_[X + 2, X - 2]
X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))

# 訓練局部離群因子模型
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred_train = clf.fit_predict(X_train)
y_pred_outliers = clf.fit_predict(X_outliers)

# 可視化結果
plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=20, edgecolor='k', label='Training data')
plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red', s=20, edgecolor='k', label='Outliers')
plt.legend()
plt.title('Local Outlier Factor (LOF)')
plt.show()