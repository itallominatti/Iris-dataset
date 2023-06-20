import unittest
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


class IrisLinearRegression:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = LinearRegression()

    def load_data(self):
        iris_data = pd.read_csv(self.data_path)
        self.X = iris_data[['sepal.length', 'sepal.width', 'petal.width']]
        self.y = iris_data['petal.length']

    def split_data(self, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        self.y_pred = self.model.predict(self.X_test)

    def evaluate(self):
        mse = mean_squared_error(self.y_test, self.y_pred)
        return mse

def create_iris_dataset_plot(data_path):
    iris_data = pd.read_csv(data_path)
    columns = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']
    sns.pairplot(iris_data, vars=columns, hue='variety')
    plt.show(block=False)
    plt.pause(0.001)
    plt.waitforbuttonpress()
    plt.close()



class IrisLinearRegressionTest(unittest.TestCase):
    def setUp(self):
        self.data_path = './excel/iris.csv'
        self.iris_regression = IrisLinearRegression(self.data_path)
        self.iris_regression.load_data()
        self.iris_regression.split_data()
        self.iris_regression.train_model()
        self.iris_regression.predict()

    def test_load_data(self):
        self.assertIsInstance(self.iris_regression.X, pd.DataFrame)
        self.assertIsInstance(self.iris_regression.y, pd.Series)

    def test_split_data(self):
        self.assertEqual(len(self.iris_regression.X_train), 120)
        self.assertEqual(len(self.iris_regression.X_test), 30)
        self.assertEqual(len(self.iris_regression.y_train), 120)
        self.assertEqual(len(self.iris_regression.y_test), 30)

    def test_train_model(self):
        self.assertIsInstance(self.iris_regression.model, LinearRegression)
        self.assertIsNotNone(self.iris_regression.model.coef_)
        self.assertIsNotNone(self.iris_regression.model.intercept_)

    def test_predict(self):
        self.assertIsNotNone(self.iris_regression.y_pred)

    def test_evaluate(self):
        mse = mean_squared_error(self.iris_regression.y_test, self.iris_regression.y_pred)
        result = self.iris_regression.evaluate()
        print(f'Mean Squared Error: {result}')
        self.assertAlmostEqual(result, mse, places=6)


if __name__ == '__main__':
    iris_regression = IrisLinearRegression('./excel/iris.csv')
    iris_regression.load_data()
    iris_regression.split_data()
    iris_regression.train_model()
    iris_regression.predict()
    iris_regression.evaluate()

    create_iris_dataset_plot('./excel/iris.csv')

    unittest.main()
