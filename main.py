import numpy as np

np.seterr(all='ignore')  # to suppress errors in color conversion fkts
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import skimage.io
from skimage.color import rgb2hsv, hsv2rgb
from sklearn.utils import shuffle
from sklearn.metrics import pairwise_distances_argmin
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn import datasets
from sklearn.model_selection import train_test_split


def aufgabe3():
    data_raw = skimage.io.imread("./resources/images/FFIX.jpg")
    data = np.array(data_raw)
    plt.figure(figsize=(4, 4))
    plt.imshow(data, interpolation='nearest')

    rgb_img = data
    # print(rgb_img)

    hsv_img = rgb2hsv(rgb_img)
    hue_img = hsv_img[:, :, 0]
    sat_img = hsv_img[:, :, 1]
    value_img = hsv_img[:, :, 2]

    fig, (ax0, ax1, ax2, ax3) = plt.subplots(ncols=4, figsize=(8, 2))

    ax0.imshow(rgb_img)
    ax0.set_title("RGB image")
    ax0.axis('off')
    ax1.imshow(hue_img, cmap='hsv')
    ax1.set_title("Hue channel")
    ax1.axis('off')
    ax2.imshow(sat_img)
    ax2.set_title('Saturation\nChannel')
    ax2.axis('off')
    ax3.imshow(value_img)
    ax3.set_title("Value channel")
    ax3.axis('off')

    fig.tight_layout()

    # 227,227,3 -> (227*227,3)
    print(hsv_img.shape)
    hsv_matrix = hsv_img.reshape(720 * 1280, 3)
    # print("resize: \n", hsv_matrix)

    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto")
    kmeans.fit(hsv_matrix)
    print(kmeans)
    # print("labels: ", kmeans.labels_)

    labels = kmeans.predict(hsv_matrix)

    # print("\ncluster_centers:\n", kmeans.cluster_centers_)

    gm = GaussianMixture(n_components=2, random_state=0)
    gm.fit(hsv_matrix)
    labels_gm = gm.predict(hsv_matrix)
    print(gm)

    test_random = shuffle(hsv_matrix, random_state=0, n_samples=64)

    hsv_to_rgb_matrix = hsv2rgb(hsv_matrix)
    labels_random = pairwise_distances_argmin(test_random, hsv_to_rgb_matrix, axis=0)
    labels_random_gm = pairwise_distances_argmin(test_random, hsv_to_rgb_matrix, axis=0)

    w, h, d = original_shape = tuple(rgb_img.shape)

    def recreate_image(codebook, labels, w, h):
        """Recreate the (compressed) image from the code book & labels"""
        return codebook[labels].reshape(w, h, -1)

    fig, (ax0, ax1, ax2, ax3) = plt.subplots(ncols=4, figsize=(10, 10))

    ax0.imshow(rgb_img)
    ax0.set_title("RGB image")
    ax0.axis('off')

    ax1.imshow(recreate_image(test_random, labels_random, w, h))
    ax1.set_title("HSV Image Random")
    ax1.axis('off')

    ax2.imshow(recreate_image(test_random, labels_random_gm, w, h))
    ax2.set_title("Gaussian Image Random")
    ax2.axis('off')

    ax3.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))
    ax3.set_title("HSV Image Cluster_Centers")
    ax3.axis('off')

    plt.show()


def aufgabe5():
    # Error Correction Algorithm

    s = [((0, 2), 1), ((1, 1), 1), ((1, 2.5), 1), ((2, 0), 0), ((3, 0.5), 0)]
    w = [0, 0, 0]
    loss = 0
    for i in range(len(s)):
        c = s[i][1]
        input0 = 1
        input1 = s[i][0][0]
        input2 = s[i][0][1]

        # calculation for o
        sigma = input0 * w[0] + input1 * w[1] + input2 * w[2]

        if sigma > 0:
            o = 1
        else:
            o = 0

        loss += (c - sigma) ** 2

        w[0] = w[0] + (c - o) * input0
        w[1] = w[1] + (c - o) * input1
        w[2] = w[2] + (c - o) * input2

    print(loss)

    # Grenzen des Plots definieren
    x_min, x_max = -0.5, 3.5
    y_min, y_max = -0.5, 3.0

    # Gitter für den Plot erzeugen
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    # Vorhersage für jedes Gitterpunkt berechnen
    Z = np.zeros(xx.shape)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            x1 = xx[i, j]
            x2 = yy[i, j]
            sigma = w[0] + w[1] * x1 + w[2] * x2
            if sigma > 0:
                Z[i, j] = 1

    # Plotten
    plt.figure()
    plt.scatter([x[0][0] for x in s[:3]], [x[0][1] for x in s[:3]], color='blue', label='Klasse 1')
    plt.scatter([x[0][0] for x in s[3:]], [x[0][1] for x in s[3:]], color='red', label='Klasse 2')
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.coolwarm)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()

    # Gradiant Desecent algorithm

    s2 = [((0, 2), 1), ((1, 1), 1), ((1, 2.5), 1), ((2, 0), 0), ((3, 0.5), 0)]

    w_new = [0, 0, 0]

    grad = np.zeros_like(w)
    for i in range(len(s)):
        c = s[i][1]
        input0 = 1
        input1 = s2[i][0][0]
        input2 = s2[i][0][1]
        sigma = input0 * w_new[0] + input1 * w_new[1] + input2 * w_new[2]
        grad[0] += -2 * (c - sigma) * input0
        grad[1] += -2 * (c - sigma) * input1
        grad[2] += -2 * (c - sigma) * input2

    print(grad)


if __name__ == '__main__':
    aufgabe3()
    aufgabe5()
