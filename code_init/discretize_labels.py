import numpy as np
import matplotlib.pyplot as plt


def get_interval(y, n_intervals, uncertainty):
    interval = []
    interval_uncertainty = []
    # unc = np.array([1 + uncertainty, 1 - uncertainty])
    unc = np.array([1 / (1 - uncertainty), 1 / (1 + uncertainty)])
    for i in range(n_intervals):
        bound = [i / n_intervals, (i + 1) / n_intervals]
        bound[0] = np.percentile(y, bound[0] * 100)
        bound[1] = np.percentile(y, bound[1] * 100)
        interval.append(bound)
        bound *= unc
        interval_uncertainty.append(list(bound))
    return interval, interval_uncertainty


def figure_plot(sample, bounds, train_intervals, test_intervals=None):
    f, ax = plt.subplots(1, 1, figsize=(20, 5))

    IQR = np.percentile(sample, 75) - np.percentile(sample, 25)
    bin_width = 2 * IQR * np.power(len(sample), -1 / 3)
    num_bins = int((np.max(sample) - np.min(sample)) / bin_width)

    d = ax.hist(sample, bins=num_bins, alpha=0.75)

    c = 0
    for interval in bounds:
        if c == 0:
            ax.axvline(x=interval[0], color="black", linestyle="-", label="interval")
        else:
            ax.axvline(x=interval[0], color="black", linestyle="-")

        ax.axvline(x=interval[1], color="black", linestyle="-")
        c += 1
    if test_intervals is not None:
        c = 0
        for interval in test_intervals:
            if c == 0:
                ax.fill_betweenx(
                    [0, plt.gca().get_ylim()[1]],
                    interval[0],
                    interval[1],
                    color="g",
                    alpha=0.15,
                    linewidth=2.5,
                    linestyle="--",
                    label="test selection",
                )
            else:
                ax.fill_betweenx(
                    [0, plt.gca().get_ylim()[1]],
                    interval[0],
                    interval[1],
                    color="g",
                    alpha=0.15,
                    linewidth=2.5,
                    linestyle="--",
                )
            c += 1
    c = 0
    for interval in train_intervals:
        if c == 0:
            ax.fill_betweenx(
                [0, plt.gca().get_ylim()[1]],
                interval[0],
                interval[1],
                color="r",
                alpha=0.15,
                linewidth=2.5,
                linestyle="--",
                label="training selection",
            )
        else:
            ax.fill_betweenx(
                [0, plt.gca().get_ylim()[1]],
                interval[0],
                interval[1],
                color="r",
                alpha=0.15,
                linewidth=2.5,
                linestyle="--",
            )
        c += 1

    ax.set_ylim([0, d[0].max() * 1.05])
    ax.set_xlim([0, d[1].max() * 1.05])
    ax.set_xlabel("dynamic range of the labels")
    ax.set_ylabel("number of samples")
    ax.legend(loc="upper right")
    plt.show()


def get_label(y, bound, labels):
    y_label = []
    for i in range(len(y)):
        for j in range(len(bound)):
            if y[i] >= bound[j][0] and y[i] < bound[j][1]:
                y_label.append(labels[j])
                break
    return np.array(y_label)


class DiscretizeLabels:
    def __init__(
        self, n_intervals, uncertainty_train, uncertainty_test=0, verbose=True
    ):
        self.n_intervals = n_intervals
        self.uncertainty_train = uncertainty_train
        self.uncertainty_test = uncertainty_test
        self.verbose = verbose
        self.y = None
        self.labels = None
        self.bound = None
        self.bound_train = None
        self.bound_test = None
        self.fitted = False

    def check(self):
        for b in self.bound:
            if b[0] >= b[1]:
                raise ValueError("The lower bound is larger than the upper bound")
        for b in self.bound_train:
            if b[0] >= b[1]:
                raise ValueError("The lower bound is larger than the upper bound")
        for b in self.bound_test:
            if b[0] >= b[1]:
                raise ValueError("The lower bound is larger than the upper bound")
        return True

    def fit(self, y):
        self.y = y
        self.bound, self.bound_train = get_interval(
            self.y, self.n_intervals, self.uncertainty_train
        )
        if self.uncertainty_test != self.uncertainty_train:
            _, self.bound_test = get_interval(
                self.y, self.n_intervals, self.uncertainty_test
            )
        else:
            self.bound_test = self.bound_train
        self.fitted = True
        self.labels = [
            f"{chr(65+i)}_{self.bound[i][0]:.3f}to{self.bound[i][1]:.3f}"
            for i in range(len(self.bound))
        ]

        if self.check() and self.verbose:
            print("Training set:")
            for i, n in enumerate(self.labels):
                print(
                    f"{n}: {np.sum((self.y >= self.bound_train[i][0]) & (self.y < self.bound_train[i][1]))}"
                )
            print("\nTest set:")
            for i, n in enumerate(self.labels):
                print(
                    f"{n}: {np.sum((self.y >= self.bound_test[i][0]) & (self.y < self.bound_test[i][1]))}"
                )

    def plot(self):
        if self.fitted:
            figure_plot(self.y, self.bound, self.bound_train, self.bound_test)
        else:
            print("Please fit the data first")

    def transform(self, y, train_index, test_index):
        if not self.fitted:
            raise ValueError("Please fit the data first")
        else:
            y_train = np.zeros(len(train_index))
            y_test = np.zeros(len(test_index))
            y_train = get_label(y[train_index], self.bound_train, self.labels)
            y_test = get_label(y[test_index], self.bound_test, self.labels)
            return y_train, y_test
