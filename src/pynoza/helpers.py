import matplotlib.pyplot as plt


class PlotAndWait:
    def __init__(self, *args, **kwargs):
        self.wait_for_enter_keypress = kwargs.pop("wait_for_enter_keypress", True)
        self.new_figure = kwargs.pop("new_figure", False)
        plt.ion()
        if self.new_figure:
            self.fig = plt.figure(*args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        plt.pause(0.0001)
        plt.show()
        if self.wait_for_enter_keypress:
            input("[Enter] to continue...")


if __name__ == "__main__":
    with PlotAndWait() as paw:
        paw.fig.add_subplot(1, 2, 1)
        plt.plot(1, 1, "x")
        paw.fig.add_subplot(1, 2, 2)
        plt.plot(1, 1, "o")
