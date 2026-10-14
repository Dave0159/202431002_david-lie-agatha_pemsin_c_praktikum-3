# Praktikum 3 - Decision Tree dan Regresi Linear

## 1. Decision Tree

### a. Import library dan load dataset

```python
#202431002_david Lie Agatha
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

```python
#202431002_david Lie Agatha
df = pd.read_csv('iris-flower.csv', sep=",")
df
```

```python
#202431002_david Lie Agatha
df.info()
```

```python
#202431002_david Lie Agatha
df['Id'].unique()
```

```python
#202431002_david Lie Agatha
df['Species'].unique()
```

### b. Preprocessing

```python
#202431002_david Lie Agatha
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])
print(df['Species'].unique())

for i, label in enumerate(le.classes_):
    print(f"{i}={label}")
```

```python
#202431002_david Lie Agatha
df
```

### c. Pemodelan Decision Tree

```python
#202431002_david Lie Agatha
x = df.drop(['Id', 'Species'], axis=1)
x
```

```python
#202431002_david Lie Agatha
y = df['Species']
y
```

```python
#202431002_david Lie Agatha
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
```

```python
#202431002_david Lie Agatha
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()
classifier.fit(x_train, y_train)
```

```python
#202431002_david Lie Agatha
from sklearn import tree

plt.subplots(figsize=(15,10))
tree.plot_tree(classifier, fontsize=10)
plt.show()
```

```python
#202431002_david Lie Agatha
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
classifier.fit(x_train, y_train)
```

```python
#202431002_david Lie Agatha
from sklearn import tree

plt.subplots(figsize=(15,10))
tree.plot_tree(classifier, fontsize=10)
plt.show()
```

```python
#202431002_david Lie Agatha
depth = classifier.get_depth()
depth
```

### d. Evaluasi akurasi

```python
#202431002_david Lie Agatha
from sklearn.metrics import accuracy_score

y_pred = classifier.predict(x_test)
akurasi = accuracy_score(y_test, y_pred)
print(akurasi)
```

```python
#202431002_david Lie Agatha
y_pred = classifier.predict([[3.3, 5, 7, 0.7]])
print(y_pred)
```

---

## 2. Regresi Linear

### a. Install library

```python
pip install scikit-learn
```

### b. Import library dan data awal

```python
#202431002_david Lie Agatha
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
```

```python
#202431002_david Lie Agatha
df = pd.DataFrame([[153,63], [174,81], [136,56],
                   [186,91], [128,47], [136,57],
                   [179,76], [163,72], [131,48]])

df.columns=["x","y"]
```

```python
#202431002_david Lie Agatha
df.corr()
```

### c. Latih model regresi

```python
#202431002_david Lie Agatha
# MEMBAGI DATA
x_train = df['x'].values[:, np.newaxis]
y_train = df['y'].values

x_train
```

```python
#202431002_david Lie Agatha
y_train
```

```python
#202431002_david Lie Agatha
lm = LinearRegression()
lm.fit(x_train, y_train)
```

```python
#202431002_david Lie Agatha
x_test = [[170], [160], [180], [171], [150]]
p = lm.predict(x_test)

p
```

```python
#202431002_david Lie Agatha
print("coefficient :", lm.coef_)
print("intercept :", lm.intercept_)
```

```python
#202431002_david Lie Agatha
pb = lm.predict(x_train)
dfc = pd.DataFrame({"x": df["x"], "y": pb})
plt.scatter(df['x'], df['y'])
plt.plot(dfc['x'], dfc['y'], color="magenta", linewidth=1)
plt.xlabel('Nilai X')
plt.ylabel('Nilai Y')
```

### d. Regresi pada dataset botak.csv

```python
#202431002_david Lie Agatha
df = pd.read_csv('botak.csv')
df
```

```python
#202431002_david Lie Agatha
import seaborn as sns

num_df = df.select_dtypes(include=['int64','float64'])

plt.figure(figsize=(10,6))
sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm")
plt.show()
```

```python
#202431002_david Lie Agatha
db_clean = df.dropna(subset=["umur", "gaji", "tinggi", "berat", "botak_prob"])

x = db_clean[["umur", "gaji", "tinggi", "berat"]]
y = db_clean["botak_prob"]
```

```python
#202431002_david Lie Agatha
#memisah data menjadi data Latih dan data test
from sklearn.model_selection import train_test_split as tts

x_train, x_test, y_train, y_test = tts(x, y, test_size=0.2, random_state=0)
```

```python
#202431002_david Lie Agatha
model = LinearRegression()
model.fit(x_train, y_train)
```

```python
#202431002_david Lie Agatha
y_pred = model.predict(x_test)
```

```python
#202431002_david Lie Agatha
from sklearn.metrics import mean_squared_error, r2_score

print("Coefficient :", model.coef_)
print("Intercept :", model.intercept_)
print("MSE :", mean_squared_error(y_test, y_pred))
print("R2 Score :", r2_score(y_test, y_pred))
```

```python
#202431002_david Lie Agatha
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Nilai asli y_test")
plt.ylabel("Nilai prediksi y_pred")
plt.title("Nilai asli vs Nilai prediksi")
plt.plot([0,1],[0,1],'r--')
plt.show()
```

---

## Ringkasan

- Decision Tree: digunakan untuk klasifikasi dataset iris.
- Regresi Linear: digunakan untuk memprediksi nilai numerik.
- Evaluasi dilakukan dengan metrik yang sesuai pada masing-masing model.
