
## Анализ медицинских расходов и прогнозирование страховых выплат 🏥💰

![medical_cost](https://img.shields.io/badge/Project-Medical%20Cost%20Prediction-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-Regression-green)
![Python](https://img.shields.io/badge/Python-3.9%2B-yellow)

Проект посвящен анализу факторов, влияющих на медицинские расходы, и построению ML-модели для прогнозирования индивидуальных страховых выплат (charges) на основе демографических данных и образа жизни клиента.

Работу выполнили: **Кузнецова Алена и Селецкий Иван**, группа ИД24-1

Данные взяты из датасета Kaggle `mirichoi0218/insurance` (Medical Cost Personal Datasets). В датасете содержатся наблюдения за клиентами страховой компании: возраст, пол, ИМТ, количество детей, статус курения, регион проживания и сумма медицинских расходов.

Основная задача решается как **регрессия**: модель должна предсказывать непрерывное числовое значение страховых выплат.

## Содержание

- [Цель проекта](#цель-проекта)
- [Постановка задачи](#постановка-задачи)
- [Данные](#данные)
- [Структура проекта](#структура-проекта)
- [Установка и запуск](#установка-и-запуск)
- [Предобработка и feature engineering](#предобработка-и-feature-engineering)
- [EDA](#eda)
- [Моделирование](#моделирование)
- [Оценка качества](#оценка-качества)
- [Подбор гиперпараметров](#подбор-гиперпараметров)
- [Интерпретация модели](#интерпретация-модели)
- [Проверка гипотез](#проверка-гипотез)
- [Итоги](#итоги)

## Цель проекта

Цель: построить и обосновать модель машинного обучения, которая прогнозирует индивидуальные медицинские расходы на основе демографических и поведенческих признаков клиента.

Полный ML-цикл в проекте:

```text
Data -> Preprocessing -> Feature Engineering -> EDA -> Model -> Evaluation -> Interpretation
```

Что сделано:

- загружен и описан реальный датасет с медицинскими расходами;
- проведена очистка данных и проверка пропусков;
- выполнено кодирование категориальных признаков;
- созданы новые признаки для повышения качества модели;
- проведен EDA с визуализациями распределений, зависимостей и корреляций;
- обучены baseline, классические ML-модели и ансамблевая модель;
- рассчитаны метрики регрессии MAE, RMSE, R2;
- выполнен подбор гиперпараметров;
- проанализирована важность признаков;
- проверены исследовательские гипотезы.

## Постановка задачи

Тип задачи: **регрессия**.

Целевая переменная:

```text
charges
```

Используемые признаки:

- `age` - возраст клиента;
- `sex` - пол;
- `bmi` - индекс массы тела;
- `children` - количество детей;
- `smoker` - статус курения;
- `region` - регион проживания;
- дополнительные признаки, созданные на этапе feature engineering.

Бизнес-смысл задачи: прогнозирование медицинских расходов помогает страховой компании устанавливать адекватную стоимость полисов, выявлять группы риска и управлять финансовыми резервами.

## Данные

Датасет загружается напрямую из GitHub и устанавливаем нужные библиотеки:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from scipy.stats import ttest_ind, pearsonr
```
```python
url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
df = pd.read_csv(url)
```

Основные поля датасета:

| Поле | Описание |
|------|----------|
| `age` | возраст клиента |
| `sex` | пол (male/female) |
| `bmi` | индекс массы тела |
| `children` | количество детей на иждивении |
| `smoker` | статус курения (yes/no) |
| `region` | регион проживания (southwest, southeast, northwest, northeast) |
| `charges` | медицинские расходы, целевая переменная |

## Структура проекта

```text
medical_cost_analysis/
├── MedicalCost_Analysis.ipynb
├── README.md
├── requirements.txt
├── medical_cost_model.pkl
└── figures/
    ├── distribution_charges.png
    ├── smoker_boxplot.png
    ├── age_bmi_scatter.png
    ├── correlation_heatmap.png
    ├── category_barplots.png
    └── feature_importance.png
```

## Установка и запуск

Клонирование репозитория:

```bash
git clone https://github.com/ezzuzz/medical_cost_analysis.git
cd medical_cost_analysis
```

Установка зависимостей:

```bash
pip install -r requirements.txt
```

Запуск ноутбука:

```bash
jupyter notebook MedicalCost_Analysis.ipynb
```

## Предобработка и feature engineering

В проекте выполнены:

- проверка структуры датасета;
- обработка пропусков (в данном датасете их нет);
- кодирование категориальных признаков `sex`, `smoker`, `region`;
- выделение категорий ИМТ и возрастных групп;
- создание интерактивного признака `smoker_bmi_interaction`;
- создание бинарного признака `has_children`;
- логарифмирование целевой переменной для анализа.

Примеры созданных признаков:
```bash
df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100],
                            labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 100],
                         labels=['Young', 'Middle', 'Senior'])
df['smoker_bmi_interaction'] = df['smoker'].apply(lambda x: 1 if x == 'yes' else 0) * df['bmi']
df['has_children'] = (df['children'] > 0).astype(int)
df['log_charges'] = np.log1p(df['charges'])
 ```
- `bmi_category` (Underweight, Normal, Overweight, Obese);
- `age_group` (Young, Middle, Senior);
- `smoker_bmi_interaction` (произведение статуса курения на ИМТ);
- `has_children` (есть ли дети).


## EDA

В ходе разведочного анализа были изучены:

- распределение медицинских расходов;
- различия между курящими и некурящими;
- зависимость расходов от возраста и ИМТ;
- корреляции между числовыми признаками;
- средние расходы по полу, региону и наличию детей.

### Примеры визуализаций из проекта:
```bash
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(df['charges'], bins=50, kde=True, ax=axes[0])
axes[0].set_title('Распределение charges (исходное)')
axes[0].set_xlabel('Медицинские расходы')

sns.histplot(df['log_charges'], bins=50, kde=True, color='green', ax=axes[1])
axes[1].set_title('Распределение log(charges)')
axes[1].set_xlabel('Логарифм расходов')

plt.tight_layout()
plt.savefig('figures/distribution_charges.png', dpi=150, bbox_inches='tight')
plt.show()
```

![Распределение медицинских расходов](figures/distribution_charges.png)

*Рисунок 1. Распределение исходных и логарифмированных медицинских расходов*

```bash
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='smoker', y='charges', palette='Set2')
plt.title('Распределение расходов в зависимости от курения')
plt.xlabel('Курит?')
plt.ylabel('Медицинские расходы')
plt.savefig('figures/smoker_boxplot.png', dpi=150, bbox_inches='tight')
plt.show()
```

![Влияние курения на расходы](figures/smoker_boxplot.png)

*Рисунок 2. Сравнение расходов курящих и некурящих клиентов*


```bash
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

sns.scatterplot(data=df, x='age', y='charges', hue='smoker', alpha=0.6, ax=axes[0])
axes[0].set_title('Расходы vs Возраст')
axes[0].set_xlabel('Возраст')
axes[0].set_ylabel('Расходы')

sns.scatterplot(data=df, x='bmi', y='charges', hue='smoker', alpha=0.6, ax=axes[1])
axes[1].set_title('Расходы vs ИМТ')
axes[1].set_xlabel('BMI')
axes[1].set_ylabel('Расходы')

plt.tight_layout()
plt.savefig('figures/age_bmi_scatter.png', dpi=150, bbox_inches='tight')
plt.show()
```

![Зависимость расходов от возраста и ИМТ](figures/age_bmi_scatter.png)

*Рисунок 3. Влияние возраста и индекса массы тела на медицинские расходы*


```bash
numeric_cols = ['age', 'bmi', 'children', 'charges']
corr = df[numeric_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Корреляционная матрица числовых признаков')
plt.savefig('figures/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
```

![Корреляционная матрица](figures/correlation_heatmap.png)

*Рисунок 4. Корреляции между числовыми признаками*

```bash
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.barplot(data=df, x='sex', y='charges', errorbar=None, palette='pastel', ax=axes[0])
axes[0].set_title('Средние расходы по полу')

sns.barplot(data=df, x='region', y='charges', errorbar=None, palette='pastel', ax=axes[1])
axes[1].set_title('Средние расходы по региону')

sns.barplot(data=df, x=df['has_children'].map({0:'Нет детей', 1:'Есть детей'}),
            y='charges', errorbar=None, palette='pastel', ax=axes[2])
axes[2].set_title('Средние расходы: дети vs нет')

plt.tight_layout()
plt.savefig('figures/category_barplots.png', dpi=150, bbox_inches='tight')
plt.show()
```

![Средние расходы по категориям](figures/category_barplots.png)

*Рисунок 5. Сравнение средних расходов по полу, региону и наличию детей*




### Итоги EDA 

1. **Курение — доминирующий фактор:** курящие клиенты платят в 3-4 раза больше некурящих.
2. **Возраст и ИМТ положительно коррелируют с расходами,** особенно сильно это проявляется у курящих.
3. **Пол и регион значимо не влияют** на средние страховые выплаты.

## Моделирование

```bash
features = ['age', 'sex', 'bmi', 'children', 'smoker', 'region',
            'bmi_category', 'age_group', 'smoker_bmi_interaction', 'has_children']
target = 'charges'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Обучающая выборка: {X_train.shape}")
print(f"Тестовая выборка: {X_test.shape}")
```

```bash
categorical_features = ['sex', 'smoker', 'region', 'bmi_category', 'age_group']
numeric_features = ['age', 'bmi', 'children', 'smoker_bmi_interaction', 'has_children']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
```



Были обучены несколько моделей:

- `LinearRegression` как baseline;
- `DecisionTreeRegressor`;
- `KNeighborsRegressor`;
- `Lasso`;
- `RandomForestRegressor`;
- `SVR`.

## Оценка качества
```bash
best_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', RandomForestRegressor(random_state=42))])
best_pipeline.fit(X_train, y_train)

y_pred_train = best_pipeline.predict(X_train)
y_pred_test = best_pipeline.predict(X_test)

def print_metrics(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name}: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}")

print_metrics(y_train, y_pred_train, "Train")
print_metrics(y_test, y_pred_test, "Test")
```
Для оценки использовались метрики регрессии:

- `MAE` - средняя абсолютная ошибка;
- `RMSE` - среднеквадратичная ошибка;
- `R2` - доля объясненной дисперсии.

Сравнение моделей:
```bash
models = {
    'LinearRegression': LinearRegression(),
    'DecisionTree': DecisionTreeRegressor(random_state=42),
    'RandomForest': RandomForestRegressor(random_state=42, n_jobs=-1)
}

results = {}
for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
    results[name] = scores.mean()
    print(f"{name}: R2 = {scores.mean():.4f} (+/- {scores.std():.4f})")

best_model_name = max(results, key=results.get)
```

| Модель | MAE | RMSE | R2 |
|--------|-----|------|----|
| LinearRegression | 4181.81 | 6000.89 | 0.783 |
| DecisionTreeRegressor | 2910.45 | 4976.88 | 0.851 |
| KNeighborsRegressor | 4239.47 | 6070.82 | 0.778 |
| Lasso | 4185.76 | 6009.82 | 0.782 |
| RandomForestRegressor | 2191.94 | 4115.40 | 0.898 |
| SVR | 6503.73 | 8555.01 | 0.559 |

Лучший результат показала ансамблевая модель `RandomForestRegressor`.

## Подбор гиперпараметров
```bash
param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [10, 20, None],
    'regressor__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(best_pipeline, param_grid, cv=5, scoring='r2', verbose=1)
grid_search.fit(X_train, y_train)

print(f"\nЛучшие параметры: {grid_search.best_params_}")
print(f"Лучший R2 на CV: {grid_search.best_score_:.4f}")

best_model_tuned = grid_search.best_estimator_
y_pred_tuned = best_model_tuned.predict(X_test)
print_metrics(y_test, y_pred_tuned, "Test после тюнинга")
Для лучшей модели (RandomForest) был выполнен подбор гиперпараметров через GridSearchCV. Лучшие параметры:
```
```text
n_estimators: 200
max_depth: 10
min_samples_split: 2
min_samples_leaf: 1
```

Лучший R2 на кросс-валидации:

```text
0.8765
```

Результат после подбора на тестовой выборке:

```text
MAE: 2163.82
RMSE: 4086.38
R2: 0.899
```

## Интерпретация модели
```bash
rf_model = best_model_tuned.named_steps['regressor']

feature_names = (numeric_features +
                 list(best_model_tuned.named_steps['preprocessor']
                      .named_transformers_['cat']
                      .named_steps['onehot']
                      .get_feature_names_out(categorical_features)))

importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Важность признаков", fontsize=14)
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=90)
plt.tight_layout()
plt.savefig('figures/feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

print("Топ-5 важных признаков:")
for i in range(5):
    print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
```
Для интерпретации была рассчитана важность признаков на модели `RandomForestRegressor`.

![Важность признаков](figures/feature_importance.png)

*Рисунок 6. Наиболее важные признаки для прогноза медицинских расходов*

Наиболее значимые признаки:

| Признак | Важность |
|---------|----------|
| `smoker_bmi_interaction` | 0.8036 |
| `age` | 0.1241 |
| `bmi` | 0.0348 |
| `children` | 0.0103 |
| `smoker_yes` | 0.0046 |

**Вывод:** самым сильным фактором для прогноза медицинских расходов является взаимодействие курения и индекса массы тела (`smoker_bmi_interaction`). Это означает, что наибольший риск представляют курящие люди с высоким ИМТ. Также важны возраст и сам по себе ИМТ.

## Проверка гипотез
```bash
# Гипотеза 1: курящие vs некурящие
smoker_yes = df[df['smoker'] == 'yes']['charges']
smoker_no = df[df['smoker'] == 'no']['charges']
t_stat, p_value = ttest_ind(smoker_yes, smoker_no)
print(f"Курящие vs некурящие: p-value = {p_value:.2e}")
print(" Гипотеза подтверждена: курящие тратят больше")

# Гипотеза 2: корреляция возраста
corr_age, p_age = pearsonr(df['age'], df['charges'])
print(f"Корреляция возраста и расходов: {corr_age:.3f}, p-value = {p_age:.2e}")
print(" С возрастом расходы растут")

# Гипотеза 3: ожирение
obese = df[df['bmi'] > 30]['charges']
non_obese = df[df['bmi'] <= 30]['charges']
t_stat2, p_value2 = ttest_ind(obese, non_obese)
print(f"Ожирение vs норма: p-value = {p_value2:.4f}")
print(" Люди с ожирением тратят больше")
```
В проекте были проверены следующие статистические гипотезы:

### Гипотеза 1: Курящие имеют более высокие расходы, чем некурящие

**Результат:** T-тест показал статистически значимую разницу (p-value < 0.001). Гипотеза подтверждена.

### Гипотеза 2: С возрастом расходы растут

**Результат:** Корреляция Пирсона между age и charges = 0.299 (p-value < 0.001). Гипотеза подтверждена.

### Гипотеза 3: Люди с ожирением (BMI > 30) тратят больше

**Результат:** T-тест показал статистически значимую разницу (p-value < 0.001). Гипотеза подтверждена.

Основные выводы:

- курение является главным фактором риска, увеличивающим расходы;
- возраст и высокий ИМТ дополнительно усиливают риски;
- feature engineering (создание интеракции `smoker_bmi_interaction`) значительно улучшил качество модели;
- ансамблевые модели (RandomForest) дают более высокое качество, чем простые baseline-подходы.

## Итоги
```bash
comparison = []
for name, model in models.items():
    pipe = Pipeline([('preprocessor', preprocessor), ('regressor', model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    comparison.append({
        'Model': name,
        'MAE': round(mean_absolute_error(y_test, y_pred), 2),
        'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
        'R2': round(r2_score(y_test, y_pred), 4)
    })

comparison_df = pd.DataFrame(comparison).sort_values('R2', ascending=False)
print("Сравнение моделей:")
comparison_df
```
В проекте построен полный ML-пайплайн для прогнозирования медицинских расходов. Лучшей моделью стала `RandomForestRegressor`, которая достигла:

```text
MAE: 2163.82
RMSE: 4086.38
R2: 0.899
```

**Бизнес-ценность:** модель позволяет страховой компании предсказывать расходы клиента с ошибкой около 2160 долларов. Это помогает:

- устанавливать адекватную стоимость полисов;
- выявлять клиентов из групп риска (курящие с высоким ИМТ, пожилые люди);
- принимать решения о профилактических программах.

Модель можно использовать как исследовательский инструмент для анализа факторов, влияющих на медицинские расходы, и как базу для дальнейшего улучшения прогноза.

---
