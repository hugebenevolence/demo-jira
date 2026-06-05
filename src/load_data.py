from sklearn.datasets import fetch_california_housing


def load_housing_data():
    data = fetch_california_housing(as_frame=True)
    return data.data, data.target
