import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, MinMaxScaler

RANDOM_STATE = 555

class GetFinalTable:
    """класс в конструкторе заполняет финальную сводную таблицу"""

    pivot_table = pd.DataFrame(
        columns=['модель', 'RMSE', 'MAE на train при cross validation', 'MAE на test'])

    def __init__(self, name_model, rmse, mae_train, mae_test=''):
        self.model = name_model
        self.rmse = rmse
        self.mae_train = mae_train
        self.mae_test = mae_test

        GetFinalTable.pivot_table.loc[len(GetFinalTable.pivot_table)] = \
            [self.model, self.rmse, self.mae_train, self.mae_test]


class ModelAnalyzer(GetFinalTable):
    """
    класс в конструкторе хранит списки категориальных и других переменных
    наследуется от GetFinalTable, так как заполняет его атрибут pivot_table
    в методе choosing_best_model формируется и возвращается предсказание
    """

    def __init__(self):
        self.categorical_cols = []
        self.other_cols = []

    def choosing_best_model(self, model, X, y, param_grid, cv=6, n_iter=500, rs=RANDOM_STATE, polynomial_degree=1,
                            standardize=False, X_test=False, y_test=False, stack=False):
        """
        метод формирует пайплайн, передаёт его RandomizedSearchCV, по сетке находит лучшие гиперпараметры,
        возвращает предсказания и экземпляр класса GetFinalTable, который дописывает в конец сводной таблицы,
        содержащей название алгоритма с лучшими параметрами и замеры метрик
        """


        if X_test is not False:

            y_pred_Xtest = model.predict(X_test)

            rmse = mean_squared_error(y_test, np.abs(y_pred_Xtest), squared=False)
            mae = mean_absolute_error(y_test, np.abs(y_pred_Xtest))

            table_row = GetFinalTable(str(model[-1]), rmse, '', mae)

            return table_row, y_pred_Xtest
        else:
            if standardize:
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), self.categorical_cols),
                        ('num', MinMaxScaler(), self.other_cols)
                    ])
            else:
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), self.categorical_cols),
                        ('num', 'passthrough', self.other_cols)
                    ])

            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('poly', PolynomialFeatures(degree=polynomial_degree)),
                ('regressor', model)
            ])

            grid = RandomizedSearchCV(pipeline, param_distributions=param_grid, cv=cv,
                                      scoring='neg_mean_absolute_error',
                                      refit=True, n_iter=n_iter, random_state=rs, n_jobs=-1)

            search = grid.fit(X, y)
            print('---' * 10, f'\nподобранные параметры:\n{search.best_params_}\n', '---' * 10)
            pipeline.set_params(**search.best_params_)
            pipeline.fit(X, y)

            y_pred = pipeline.predict(X)

            rmse = mean_squared_error(y, np.abs(y_pred), squared=False)

            table_row = GetFinalTable(str(pipeline[-1]), rmse, -search.best_score_, '')

            return pipeline, table_row, y_pred


if __name__ == '__main__':
    pass

