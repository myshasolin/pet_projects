import pandas as pd
import numpy as np

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm

import shap
from colorama import Fore, Style

import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 555


def get_hist(col, bins=20):
    """
    функция строит гистограмму распределения на заданное количество бинов
    дополнительно отрисовываются 3 линии на распределении - среднее, медиана и мода
    """

    mean_ = col.mean()
    median_ = col.median()
    best_ = col.value_counts().idxmax()

    plt.figure(figsize=(12, 4))
    plt.title(f'Распределение в {col.name}', fontsize=16, fontweight='bold')
    sns.histplot(col, kde=True, color='#2E86C1', ec='#196F3D', bins=bins)
    plt.ylabel('количество')

    plt.axvline(x=mean_, label=f'среднее={mean_:.4f}', lw=1.5, c='#8A2BE2')
    plt.axvline(x=median_, label=f'медиана={median_:.4f}', lw=1.5, c='#2B9902')
    plt.axvline(x=best_, label=f'мода={best_:.4f}', lw=1.5, c='#D30E0E')
    plt.legend(facecolor='oldlace', edgecolor='#7B6DA5')

    plt.minorticks_on()
    plt.grid(which='major', linewidth=.5)
    plt.grid(which='minor', linewidth=.25, linestyle='--');


def get_info(df):
    """функция для вывода общей информапции"""

    print(f'{Fore.RED}{Style.BRIGHT}таблица {df.name}{Style.RESET_ALL}\n' \
          f'{Style.BRIGHT}общая информация:{Style.RESET_ALL}\n')
    if df.info() is not None:
        print(df.info())
    print(f'\n{Style.BRIGHT}размер:{Style.RESET_ALL} {df.shape}\n')
    print(f'{Style.BRIGHT}статистика:{Style.RESET_ALL}')
    display(df.describe().T)
    print(f'{Style.BRIGHT}первые/последние строки:{Style.RESET_ALL}')
    display(df.head(3), df.tail(3))
    print(f'{Style.BRIGHT}пропуски в значениях:{Style.RESET_ALL}\n')
    missing_values = df.isna().sum()
    if not missing_values.empty:
        print(missing_values)
    print(f'\n{Style.BRIGHT}явных дубликатов:{Style.RESET_ALL} {df.duplicated().sum()}')


def plot_feature_counts(df, name=False, not_title=False, plot_type='bar', rotation=0):
    """Функция для отрисовки графиков с количеством значений в каждом признаке"""

    df_ = df.copy()
    df_ = df_.drop(columns='ladle')

    if name is False:
        try:
            name = f'для {df.name}'
        except AttributeError:
            name = ''
    else:
        name = 'для ' + name
    feature_counts = df_.count()
    feature_names = feature_counts.index

    plt.figure(figsize=(12, 4))

    if plot_type == 'bar':
        plt.bar(feature_names, feature_counts, color='#96C2DF', ec='#196F3D', width=.4)
        plt.xticks(rotation=rotation)
    elif plot_type == 'box':
        sns.boxplot(df_.values, width=.5, fliersize=2., linewidth=.7)
        plt.xticks(range(len(feature_names)), feature_names, rotation=rotation)
    elif plot_type == 'violin':
        sns.violinplot(df_.values, width=.7, fliersize=2., linewidth=.7, color='#96C2DF')
        plt.xticks(range(len(feature_names)), feature_names, rotation=rotation)
    else:
        raise ValueError(f'Некорректный тип графика: {plot_type}')

    plt.ylabel('Количество значений')
    if not_title is False:
        plt.title(f'Количество значений в каждом признаке {name}', fontsize=16, fontweight='bold')
    else:
        pass
    plt.minorticks_on()
    plt.grid(which='major', linewidth=.5)
    plt.grid(which='minor', linewidth=.25, linestyle='--');


def time_check(data):
    """функция проверки распределения времени в столбце, его начала и конца"""

    print(f'{Style.BRIGHT}столбец {data.name}{Style.RESET_ALL}')
    monotony_control = data.is_monotonic_increasing
    print('даты распределены равномерно') if monotony_control else print('даты распределены неравномерно')
    print(f'начало: {data.min()}\nконец: {data.max()}')


def plot_distribution(df, up=10):
    """функция отрисовывает распределение признака с шагом в 10%"""

    plt.figure(figsize=(12, 4))
    percentiles = range(0, 101, 10)
    values = [df.quantile(i / 100) for i in percentiles]
    plt.plot(percentiles, values, marker='v', color='#2E86C1', markersize=10, markeredgecolor='#196F3D')
    plt.xlabel('процентиль')
    plt.ylabel(df.name)

    plt.xticks(percentiles, [f'{i}%' for i in percentiles])

    plt.title(f'распределение с шагом в 10% для {df.name}', fontsize=16, fontweight='bold')
    plt.minorticks_on()
    plt.grid(which='major', linewidth=.5)
    plt.grid(which='minor', linewidth=.25, linestyle='--')
    for i, value in enumerate(values):
        plt.annotate(f'{value:.2f}', xy=(percentiles[i], value), xytext=(percentiles[i] - 2.5, value + up),
                     arrowprops=dict(arrowstyle='wedge', color='#2E86C1'));


def to_snake_case(column_names):
    """
    Функция преобразует названия колонок в snake_case
    """
    snake_case_names = []
    for name in column_names:
        # Заменяем пробелы и заглавные буквы на символ подчеркивания и маленькие буквы
        snake_case_name = name.replace(" ", "_").lower()
        snake_case_names.append(snake_case_name)
    return snake_case_names


def return_model_for_shap(X, y, model, rs=RANDOM_STATE):
    """финкция отрисовывает SHAP-график влияния признаков на отток"""

    X_ = X.copy()
    explainer = shap.TreeExplainer(model.fit(X_, y))
    preds_train = model.predict(X_)
    shap_values_all = explainer.shap_values(X_)
    shap.summary_plot(shap_values_all, X_, plot_size=(13, 8), max_display=25)


def evaluate_feature_significance_ols(X, y, alpha=0.05):
    """
    функция выполняет оценку статистической значимости признаков по методу наименьших квадратов
    и отрисовывает столбчатый график со значением alpha
    """

    # добавляем столбец с целевой переменной в X
    X_train_with_target = X.copy()
    X_train_with_target['target'] = y

    # строим модель регрессии с помощью OLS (Ordinary Least Squares)
    model = sm.OLS(X_train_with_target['target'], X_train_with_target.drop(columns='target'))
    results = model.fit(alpha=alpha)

    # преобразовываем таблицу summary().tables[1] в Pandas DataFrame
    coef_pval_table = pd.DataFrame(results.summary().tables[1].data[1:], columns=results.summary().tables[1].data[0])
    coef_pval_table['P>|t|'] = pd.to_numeric(coef_pval_table['P>|t|'])
    coef_pval_table['is important'] = True
    coef_pval_table.loc[coef_pval_table['P>|t|'] > 0.05, 'is important'] = False
    OLS_sorted_table = coef_pval_table.sort_values(by='P>|t|')
    OLS_sorted_table = OLS_sorted_table.rename(columns={OLS_sorted_table.columns[0]: "features"})

    display(results.summary().tables[0], OLS_sorted_table, results.summary().tables[2])

    plt.figure(figsize=(12, 5))
    plt.bar(OLS_sorted_table['features'], OLS_sorted_table['P>|t|'],
           color='#96C2DF', ec='#196F3D', alpha=0.7)
    plt.axhline(y=0.05, color='r', linestyle='--', label=f'{alpha=:.0%}')
    plt.legend(facecolor='oldlace', edgecolor='#7B6DA5', fontsize=22)
    plt.xticks(rotation=90)
    plt.title(f'статистический уровень значимости признаков \nпо методу наименьших квадратов',
              fontsize=16, fontweight='bold')
    plt.ylabel('p_value', fontsize=12)
    plt.minorticks_on()
    plt.grid(which='major', linewidth=.5)
    plt.grid(which='minor', linewidth=.25, linestyle='--');


def get_corr_matrix(matrix, correlation_threshold=0.3):
    "функция отрисовывает корреляционную матрицу на тепловой карте"

    sns.set(font_scale=0.75)
    sns.set_style("white")
    plt.figure(figsize = (12, 8))
    cmap = mcolors.LinearSegmentedColormap.from_list('my_palette', ['#D6E8F4', '#196F3D'])

    matrix = matrix.round(2)
    matrix[np.abs(matrix) < correlation_threshold] = 0
    mask = np.zeros_like(matrix)
    mask[np.triu_indices_from(mask)] = True

    ax = sns.heatmap(matrix, annot=True, linewidths=.3, cmap=cmap, mask=mask, annot_kws={'size': 6})
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
    plt.title('корреляционная матрица', fontweight='bold', fontsize=14);


def plot_barh(table):
    """функция отрисовывает таблицу с предсказанием MAE на тренировочной выборке"""

    plt.figure(figsize=(12, 7))

    model_names = table[table.columns[0]].tolist()
    mae_values = table[table.columns[2]].tolist()

    bars = plt.barh(range(len(model_names)), mae_values, color='#96C2DF', ec='#196F3D', alpha=0.7, height=0.5)
    plt.yticks(range(len(model_names)), model_names)
    plt.title('Результаты по значению MAE на кросс-валидации тренировочной выборки', fontweight='bold', fontsize=16)
    plt.grid(axis='y', linewidth=.5)

    for i, bar in enumerate(bars):
        plt.text(bar.get_width() - 2, bar.get_y() + bar.get_height() / 2,
                 f'MAE = {round(mae_values[i], 4)}', ha='left', va='center', fontsize=10)

    plt.axvline(x=6, label='контрольное знеачение MAE=6', lw=1.5, linestyle='--', c='#D30E0E', alpha=0.5)
    plt.legend(loc='lower right', facecolor='oldlace', edgecolor='#7B6DA5')

    plt.minorticks_on()
    plt.grid(which='major', linewidth=.5)
    plt.grid(which='minor', linewidth=.25, linestyle='--');


def model_comparison(X_train, y_train, X_test, y_test, model_prediction, strategy='mean'):
    """функция находит предсказание dummy-модели по заданной стратегии и отрисовывает столбчатый график с подписью значений"""

    dummy_clf = DummyRegressor(strategy=strategy)
    dummy_clf.fit(X_train, y_train)
    dummy_pred = dummy_clf.predict(X_test)
    dummy_prediction = (mean_squared_error(y_test, dummy_pred, squared=False), mean_absolute_error(y_test, dummy_pred))
    dummy_prediction_rounded = np.round(dummy_prediction, 3)
    model_prediction_rounded = np.round(model_prediction, 3)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x=[0], height=[model_prediction[0]], label='Модель: RMSE', alpha=0.6, color='#49A88F', ec='#2E6D57')
    ax.bar(x=[1], height=[model_prediction[1]], label='Модель: MAE', alpha=0.6, color='#FFBE5C', ec='#C47F34')
    ax.bar(x=[3], height=[dummy_prediction[0]], label='Dummy: RMSE', alpha=0.6, color='#5499C7', ec='#3167A8')
    ax.bar(x=[4], height=[dummy_prediction[1]], label='Dummy: MAE', alpha=0.6, color='#986DB2', ec='#674F8C')

    for i, value in enumerate(model_prediction_rounded):
        ax.text(i, model_prediction[i], str(value), ha='center', va='bottom')
    for i, value in enumerate(dummy_prediction_rounded):
        ax.text(i+3, dummy_prediction[i], str(value), ha='center', va='bottom')

    ax.set_xticks([0, 1, 3, 4], ['RMSE', 'MAE', 'RMSE', 'MAE'])
    plt.legend(facecolor='oldlace', edgecolor='#7B6DA5', loc='lower center')
    plt.title(f'Сравнение предсказаний модели и dummy-модели,\nстратегия: {strategy}', fontweight='bold', fontsize=16)
    plt.minorticks_on()
    plt.grid(which='major', linewidth=.5)
    plt.grid(which='minor', linewidth=.25, linestyle='--');


if __name__ == '__main__':
    pass
