import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Embedding (embedding):
    def __init__ (X, y):
        self.X = X
        self.y = y
        self.projection = None

    def plot(self):
        _, ax = plt.subplots()
        X = MinMaxScaler().fit_transform(X)

        for digit in digits.target_names:
            ax.scatter(
                *X[y == digit].T,
                marker=f"${digit}$",
                s=60,
                color=plt.cm.Dark2(digit),
                alpha=0.425,
                zorder=2,
            )
        shown_images = np.array([[1.0, 1.0]])  # just something big
        for i in range(X.shape[0]):
            # plot every digit on the embedding
            # show an annotation box for a group of digits
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.concatenate([shown_images, [X[i]]], axis=0)
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r), X[i]
            )
            imagebox.set(zorder=1)
            ax.add_artist(imagebox)

        ax.set_title(title)
        ax.axis("off")
        plt.show()

    def fit(self):
        self.projection = embedding.fit_transform(X, y)
        

    