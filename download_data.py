
from pathlib import Path
from sklearn.datasets import load_breast_cancer

# Папка проекта
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Загружаем учебный датасет
dataset = load_breast_cancer(as_frame=True)

# Сохраняем его в CSV
df = dataset.frame
file_path = DATA_DIR / "breast_cancer.csv"
df.to_csv(file_path, index=False)

print("Датасет успешно сохранён!")
print("Количество строк:", df.shape[0])
print("Количество столбцов:", df.shape[1])
print("Файл:", file_path)
