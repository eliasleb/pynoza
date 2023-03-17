import pandas as pd


def read_mikheev_data():
    data1 = pd.read_csv("data/data_mikheev_part_1.csv")

    print(data1)


if __name__ == "__main__":
    read_mikheev_data()
