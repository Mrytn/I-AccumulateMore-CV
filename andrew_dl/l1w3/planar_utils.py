import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model


def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    # np.arange(x_min, x_max, h) 和 np.arange(y_min, y_max, h) 分别生成 x 轴和 y 轴的网格点坐标。
# xx 是 x 坐标的网格，yy 是 y 坐标的网格，它们的形状都是 (num_points_x, num_points_y)。
# np.c_[xx.ravel(), yy.ravel()] 将两个一维数组按列堆叠，形成一个 (N_points, 2) 的数组，每一行是一个二维坐标点。
# 然后，将这些坐标传入训练好的 model 中进行预测，得到对应的预测类别 Z。Z 的形状是 (num_points_x * num_points_y,)，表示每个网格点的预测类别。

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    # xx.ravel() 和 yy.ravel() 将 xx 和 yy 网格从二维数组转换成一维数组。
# np.c_[xx.ravel(), yy.ravel()] 将两个一维数组/
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=np.squeeze(y), cmap=plt.cm.Spectral)


def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s


def load_planar_dataset():
    np.random.seed(1)
    m = 400  # number of examples
    N = int(m/2)  # number of points per class
    D = 2  # dimensionality
    X = np.zeros((m, D))  # data matrix where each row is a single example
    # labels vector (0 for red, 1 for blue)
    Y = np.zeros((m, 1), dtype='uint8')
    a = 4  # maximum ray of the flower
# for j in range(2)：循环两次，分别为 2 个类别（类别 0 和 类别 1）生成数据。
# ix = range(N*j, N*(j+1))：为每个类别分配 N 个样本。ix 是一个范围，表示该类别的样本索引。
# t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2：生成 N 个角度值 t，使用 np.linspace 创建均匀的角度分布，并加入一些噪声（通过 np.random.randn(N) 实现）。
# 这里 3.12 是 π 的近似值，用于模拟一个花瓣形状的波动。加入噪声使得数据更具多样性，避免过于规则。
# r = a * np.sin(4 * t) + np.random.randn(N) * 0.2：根据角度 t 计算半径 r，然后将其与噪声相加。a * np.sin(4 * t) 描述了花瓣形状的波动，+ np.random.randn(N) * 0.2 是加入的噪声，进一步增强数据的多样性。
    for j in range(2):
        # N 是每个类别的样本数，这里是 200。
        # 对于类别 0（j=0），ix 的范围是从 0 到 199，表示前 200 个样本。
        # 对于类别 1（j=1），ix 的范围是从 200 到 399，表示后 200 个样本。
        ix = range(N*j, N*(j+1))
        t = np.linspace(j*3.12, (j+1)*3.12, N) + \
            np.random.randn(N)*0.2  # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2  # radius
        # X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]：将极坐标 r 和 t 转换为笛卡尔坐标，得到 x 和 y 坐标（即 r * sin(t) 和 r * cos(t)）。这些数据点存储在 X 中。
# Y[ix] = j：为每个样本赋予标签，j 的值为 0 或 1，表示数据点所属的类别。
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T

    return X, Y


def load_extra_datasets():
    N = 200
    # 功能：生成一个“圆形”数据集，包含两个类别（通常是二分类任务），并加上噪声，使得数据不再完美地成环形。
# 参数：
# n_samples=N：生成200个样本。
# factor=.5：控制内外圆之间的半径比值，值越小，两个圆圈之间的距离越近。
# noise=.3：控制数据的噪声量，噪声越大，数据点越不规则。
# 这种数据集常用于测试模型的分类性能，尤其是处理具有非线性决策边界的情况。例如，决策树、支持向量机（SVM）等。
    noisy_circles = sklearn.datasets.make_circles(
        n_samples=N, factor=.5, noise=.3)
    # 功能：生成一个“月亮形状”的数据集，包含两个类别。数据集的形状像两个半月，并加入噪声使得数据点不再完美地沿着月亮的形状分布。
# 参数：
# n_samples=N：生成200个样本。
# noise=.2：控制噪声量，噪声越大，数据点的分布越乱。
# ：类似于圆形数据集，适用于测试具有非线性决策边界的分类算法。月亮形数据集可以很好地测试模型在处理复杂边界时的表现。
    noisy_moons = sklearn.datasets.make_moons(n_samples=N, noise=.2)
    # 功能：生成多个（通常是高斯分布的）簇（或称为“blob”）数据。数据点分布在多个簇中，每个簇代表一个类别。
# 参数：
# n_samples=N：生成200个样本。
# random_state=5：设置随机种子，以保证每次生成的数据集相同。
# n_features=2：数据是二维的。
# centers=6：生成6个簇。
# 用于生成典型的分类问题数据集，数据点分布在多个簇中，适用于测试多类别分类问题。簇与簇之间的边界通常是线性的，因此对简单的线性分类模型也很有效。
    blobs = sklearn.datasets.make_blobs(
        n_samples=N, random_state=5, n_features=2, centers=6)
    # 功能：生成两个高斯分布的类别数据，每个类别的样本由正态分布生成，通常用于测试线性分类模型。
# 参数：
# mean=None：默认情况下生成的两个类别的均值为0。
# cov=0.5：协方差，控制数据的分布宽度，值越小，数据点聚集得越紧。
# n_samples=N：生成200个样本。
# n_features=2：数据是二维的。
# n_classes=2：生成两个类别。
# shuffle=True：打乱生成的数据集。
# random_state=None：随机种子，控制数据的随机性
# 适合用来测试简单的线性分类算法（如逻辑回归、线性支持向量机等）。该数据集通常有一个明确的线性决策边界，因为每个类别是由高斯分布生成的。
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(
        mean=None, cov=0.5, n_samples=N, n_features=2, n_classes=2, shuffle=True, random_state=None)
    # 功能：生成两个完全随机分布的数据集，每个数据集包含200个样本。数据点在0到1之间均匀分布，没有任何结构。
    # 这是一个无结构的数据集，通常用于测试模型在没有明显结构或模式的情况下的表现。这样的数据集对于测试模型是否能正确处理完全随机数据非常有用。
    no_structure = np.random.rand(N, 2), np.random.rand(N, 2)

    return noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure
