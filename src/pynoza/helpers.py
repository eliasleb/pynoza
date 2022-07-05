import matplotlib.pyplot as plt


class PlotAndWait:
    def __init__(self, *args, **kwargs):
        plt.ion()
        self.fig = plt.figure(*args, **kwargs)

    def __enter__(self, subplot=None, **kwargs):
        if subplot is not None:
            plt.subplot(**kwargs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        plt.pause(0.0001)
        plt.show()

        input("[Enter] to continue...")


if __name__ == "__main__":
    with PlotAndWait() as paw:
        plt.plot(1, 1, "x")
