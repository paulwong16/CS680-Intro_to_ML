from GMM import GMM
import numpy as np
import torchvision.datasets as datasets
import matplotlib.pyplot as plt


if __name__ == '__main__':
    test_accuracy = []
    K = [i for i in range(1, 11)]
    # test K from 1 to 10
    for k_component in K:
        train_set = datasets.MNIST(root='./data/', train=True, download=True)
        gmms = []
        losses = []
        Pr_y = []
        # create gmm for each class
        for i in range(10):
            idx = train_set.targets == i
            X = train_set.data[idx].numpy().reshape((-1, 28*28)) / 255
            X += np.random.normal(0, 1, X.shape)
            Pr_y.append(len(train_set.data[idx].numpy()) / len(train_set.data))
            gmms.append(GMM(X, k_component))
            losses.append(gmms[-1].fit())
        pass
        test_set = datasets.MNIST(root='./data/', train=False, download=True)
        testX = test_set.data.numpy().reshape((-1, 28*28)) / 255
        testX += np.random.normal(0, 1, testX.shape)
        testy = test_set.targets.numpy()
        test_result = []
        for i in range(10):
            test_result.append(gmms[i].cal_density(testX) * Pr_y[i])
        test_pred = np.argmax(test_result, axis=0)
        correct = 0
        n = len(testy)
        for i in range(n):
            if testy[i] == test_pred[i]:
                correct += 1

        print('K = %i, Test accuracy: %f' % (k_component, (correct / n)))
        test_accuracy.append((correct / n))
    plt.figure()
    plt.xlabel('K')
    plt.ylabel('Test accuracy')
    plt.plot(K, test_accuracy)
    plt.show()
    pass