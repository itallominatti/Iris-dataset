import unittest
import pandas as pd
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
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
    
    def train_model(self):
        self.model.fit(self.X_train, self.y_train)
    
    def predict(self):
        self.y_pred = self.model.predict(self.X_test)
    
    def evaluate(self):
        mse = mean_squared_error(self.y_test, self.y_pred)
        return mse
    
class Plotter:
    @staticmethod
    def plot_results(y_test, y_pred):
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, color='blue', label='Valores reais vs. Previsões')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Linha de referência')
        ax.set_xlabel('Valores reais')
        ax.set_ylabel('Previsões')
        ax.set_title('Gráfico de dispersão dos valores reais vs. previsões')
        ax.legend()
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
        print(f'Erro quadrático médio: {result}')
        self.assertAlmostEqual(result, mse, places=6)


if __name__ == '__main__':
    iris_regression = IrisLinearRegression('./excel/iris.csv')
    iris_regression.load_data()
    iris_regression.split_data()
    iris_regression.train_model()
    iris_regression.predict()
    iris_regression.evaluate()
    
    plotter = Plotter()
    plotter.plot_results(iris_regression.y_test, iris_regression.y_pred)

    unittest.main()

