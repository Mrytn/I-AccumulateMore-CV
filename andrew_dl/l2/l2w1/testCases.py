import numpy as np


def compute_cost_with_regularization_test_case():
    np.random.seed(1)
    Y_assess = np.array([[1, 1, 0, 1, 0]])
    W1 = np.random.randn(2, 3)
    b1 = np.random.randn(2, 1)
    W2 = np.random.randn(3, 2)
    b2 = np.random.randn(3, 1)
    W3 = np.random.randn(1, 3)
    b3 = np.random.randn(1, 1)
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
    a3 = np.array(
        [[0.40682402,  0.01629284,  0.16722898,  0.10118111,  0.40682402]])
    return a3, Y_assess, parameters


def backward_propagation_with_regularization_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(3, 5)
    Y_assess = np.array([[1, 1, 0, 1, 0]])
    # 确保 cache 中的两个数组的形状正确，并且数组结构完整
    cache = (
        np.array([[-1.52855314,  3.32524635,  2.13994541,  2.60700654, -0.75942115],
                  [-1.98043538,  4.1600994,  0.79051021,  1.46493512, -0.45506242]]),
        np.array([[0.,  3.32524635,  2.13994541,  2.60700654,  0.],
                  [0.,  4.1600994,  0.79051021,  1.46493512,  0.]])
    )
    return X_assess, Y_assess, cache
