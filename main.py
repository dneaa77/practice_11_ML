
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =====================================
# 1. ПАПКИ И ЗАГРУЗКА ДАННЫХ
# =====================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "breast_cancer.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("=== ИНФОРМАЦИЯ О ДАННЫХ ===")
print("Размер датасета:", df.shape)
print("Пропущенных значений:", df.isnull().sum().sum())

X = df.drop(columns=["target"])
y = df["target"]

print("Количество исходных признаков:", X.shape[1])

# =====================================
# 2. РАЗДЕЛЕНИЕ ДАННЫХ
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n=== РАЗДЕЛЕНИЕ ДАННЫХ ===")
print("Обучающая выборка:", X_train.shape)
print("Тестовая выборка:", X_test.shape)

# =====================================
# 3. БАЗОВАЯ МОДЕЛЬ
# =====================================

# Масштабирование нужно и базовой модели,
# чтобы сравнение с PCA было честным.

base_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000))
])

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

base_cv_scores = cross_val_score(
    base_model,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy"
)

start = perf_counter()
base_model.fit(X_train, y_train)
base_time = perf_counter() - start

base_pred = base_model.predict(X_test)
base_accuracy = accuracy_score(y_test, base_pred)

print("\n=== БАЗОВАЯ МОДЕЛЬ ===")
print("CV Accuracy:", round(base_cv_scores.mean(), 4))
print("Test Accuracy:", round(base_accuracy, 4))
print("Время обучения:", round(base_time, 6), "сек")
print("Признаков:", X_train.shape[1])

# =====================================
# 4. МОДЕЛИ С PCA
# =====================================

results = [{
    "Модель": "Без PCA",
    "CV Accuracy": base_cv_scores.mean(),
    "Test Accuracy": base_accuracy,
    "Время (сек)": base_time,
    "Компоненты": X_train.shape[1]
}]

pca_variants = [0.80, 0.90, 0.95]

for variance in pca_variants:

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(
            n_components=variance,
            svd_solver="full"
        )),
        ("model", LogisticRegression(max_iter=2000))
    ])

    # PCA обучается внутри каждого CV-разбиения
    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy"
    )

    start = perf_counter()
    pipeline.fit(X_train, y_train)
    training_time = perf_counter() - start

    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    components = pipeline.named_steps[
        "pca"
    ].n_components_

    results.append({
        "Модель": f"PCA {int(variance * 100)}%",
        "CV Accuracy": cv_scores.mean(),
        "Test Accuracy": accuracy,
        "Время (сек)": training_time,
        "Компоненты": components
    })

    print(f"\n=== PCA {int(variance * 100)}% ===")
    print("CV Accuracy:", round(cv_scores.mean(), 4))
    print("Test Accuracy:", round(accuracy, 4))
    print("Время:", round(training_time, 6), "сек")
    print("Компонентов:", components)

# =====================================
# 5. ТАБЛИЦА СРАВНЕНИЯ
# =====================================

results_df = pd.DataFrame(results)

print("\n=== ИТОГОВАЯ ТАБЛИЦА ===")
print(results_df.to_string(index=False))

results_df.to_csv(
    RESULTS_DIR / "comparison.csv",
    index=False
)

# =====================================
# 6. ГРАФИК КАЧЕСТВА
# =====================================

plt.figure(figsize=(9, 5))

x = np.arange(len(results_df))
width = 0.35

plt.bar(
    x - width / 2,
    results_df["CV Accuracy"],
    width,
    label="Cross-validation"
)

plt.bar(
    x + width / 2,
    results_df["Test Accuracy"],
    width,
    label="Test"
)

plt.xticks(x, results_df["Модель"])
plt.ylabel("Accuracy")
plt.title("Model accuracy: PCA comparison")
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "accuracy.png",
    dpi=200
)
plt.close()

# =====================================
# 7. ГРАФИК ВРЕМЕНИ ОБУЧЕНИЯ
# =====================================

plt.figure(figsize=(9, 5))

plt.bar(
    results_df["Модель"],
    results_df["Время (сек)"]
)

plt.ylabel("Training time (seconds)")
plt.title("Training time comparison")
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "training_time.png",
    dpi=200
)
plt.close()

# =====================================
# 8. ГРАФИК КОЛИЧЕСТВА ПРИЗНАКОВ
# =====================================

plt.figure(figsize=(9, 5))

plt.bar(
    results_df["Модель"],
    results_df["Компоненты"]
)

plt.ylabel("Number of features/components")
plt.title("Dimensionality comparison")
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "components.png",
    dpi=200
)
plt.close()

# =====================================
# 9. ГРАФИК ОБЪЯСНЁННОЙ ДИСПЕРСИИ
# =====================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

pca_full = PCA()
pca_full.fit(X_scaled)

cumulative_variance = np.cumsum(
    pca_full.explained_variance_ratio_
)

plt.figure(figsize=(9, 5))

plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance,
    marker="o",
    markersize=3
)

for threshold in pca_variants:
    plt.axhline(
        y=threshold,
        linestyle="--",
        label=f"{int(threshold * 100)}%"
    )

plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA explained variance")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "explained_variance.png",
    dpi=200
)
plt.close()

# =====================================
# 10. ВЫБОР ЛУЧШЕЙ МОДЕЛИ
# =====================================

# Выбираем по CV Accuracy.
# При равенстве выбираем меньше компонент.

best = results_df.sort_values(
    by=["CV Accuracy", "Компоненты"],
    ascending=[False, True]
).iloc[0]

print("\n=== ЛУЧШАЯ МОДЕЛЬ ПО CV ===")
print("Модель:", best["Модель"])
print("CV Accuracy:", round(best["CV Accuracy"], 4))
print("Test Accuracy:", round(best["Test Accuracy"], 4))
print("Компонентов:", int(best["Компоненты"]))

print("\nГотово! Результаты сохранены в results.")
