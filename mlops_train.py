import os
import re
import csv
import joblib
import logging
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import top_k_accuracy_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

RU_ENDINGS = re.compile(r'(ами|ями|ого|его|ому|ему|ыми|ими|овать|евать|ться|тись|ать|ять|ить|еть|уть|ыть|ов|ев|ей|ам|ям|ах|ях|ом|ем|ой|ей|ий|ый|ая|яя|ое|ее|ые|ие|у|ю|а|я|о|е|ы|и|ь)$')

def stem_russian_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'[^а-яёa-z\s]', ' ', text.lower())
    return " ".join(RU_ENDINGS.sub('', w) if len(w) > 3 else w for w in text.split() if len(w) >= 3)

def normalize_macro_genre(raw_genre: str) -> str:
    first_tag = re.split(r'[,/|]', raw_genre.lower())[0].strip()
    if any(k in first_tag for k in ['фантаст', 'sci-fi', 'фэнтези', 'fantasy', 'приключен', 'adventure']): return 'фантастика и приключения'
    if any(k in first_tag for k in ['ужас', 'horror', 'триллер', 'thriller', 'детектив', 'криминал', 'crime', 'mystery']): return 'триллер, ужасы и детектив'
    if any(k in first_tag for k in ['боевик', 'action', 'военн', 'war', 'вестерн']): return 'боевик и военный'
    if any(k in first_tag for k in ['комед', 'comedy']): return 'комедия'
    if any(k in first_tag for k in ['мульт', 'анимац', 'animation', 'детск', 'семейн', 'family']): return 'анимация и семейный'
    if any(k in first_tag for k in ['документ', 'documentary', 'биограф', 'истор']): return 'документальный и история'
    if any(k in first_tag for k in ['драм', 'drama', 'мелодрам', 'romance']): return 'драма и мелодрама'
    return 'другое'

def load_dataset(csv_path="films_data.csv", limit=15000):
    texts, genres = [], []
    with open(csv_path, mode='r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if len(texts) >= limit:
                break
            raw_genre = row.get('genre', '')
            full_desc = f"{row.get('summary', '')} {row.get('plot', '')}".strip()
            if raw_genre and len(full_desc) > 80:
                clean_g = normalize_macro_genre(raw_genre)
                if clean_g != 'другое':
                    stemmed = stem_russian_text(f"{row.get('title', '')} {full_desc}")
                    if len(stemmed) > 30:
                        texts.append(stemmed)
                        genres.append(clean_g)
    return train_test_split(texts, genres, test_size=0.15, random_state=42, stratify=genres)

def run_mlflow_experiments():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("dock-assistent-classifier")
    
    logging.info("📂 Загрузка датасета для серии MLOps экспериментов...")
    X_train, X_test, y_train, y_test = load_dataset()

    # Сетка гиперпараметров (4 запуска для сравнения в UI)
    configs = [
        {"C": 0.2, "max_features": 30000},
        {"C": 0.5, "max_features": 50000},
        {"C": 0.8, "max_features": 75000},
        {"C": 1.2, "max_features": 100000},
    ]

    best_acc = 0.0
    best_run_id = None
    history_labels, history_scores = [], []

    for cfg in configs:
        run_name = f"SVC_C={cfg['C']}_feat={cfg['max_features']}"
        with mlflow.start_run(run_name=run_name) as run:
            # 1. Логируем гиперпараметры и тег
            mlflow.log_param("C", cfg["C"])
            mlflow.log_param("max_features", cfg["max_features"])
            mlflow.log_param("ngram_range", "(1, 2)")
            mlflow.set_tag("developer", "TUUYA12")
            mlflow.set_tag("model_family", "LinearSVC_Calibrated")

            # 2. Собираем и обучаем единый Pipeline (векторизатор + классификатор)
            pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(max_features=cfg["max_features"], ngram_range=(1, 2), min_df=3, sublinear_tf=True)),
                ("clf", CalibratedClassifierCV(LinearSVC(C=cfg["C"], class_weight="balanced", random_state=42), cv=3))
            ])
            pipeline.fit(X_train, y_train)

            # 3. Считаем метрики
            acc_top1 = pipeline.score(X_test, y_test)
            probs = pipeline.predict_proba(X_test)
            acc_top2 = top_k_accuracy_score(y_test, probs, k=2, labels=pipeline.classes_)

            mlflow.log_metric("test_accuracy_top1", round(acc_top1, 4))
            mlflow.log_metric("test_accuracy_top2", round(acc_top2, 4))

            history_labels.append(f"C={cfg['C']}")
            history_scores.append(acc_top2)

            # 4. Сохраняем дополнительный артефакт (график сравнения и конфиг)
            plt.figure(figsize=(6, 4))
            plt.bar(history_labels, history_scores, color="teal")
            plt.ylim(0.5, 1.0)
            plt.title("Top-2 Accuracy by Run")
            plt.ylabel("Accuracy")
            plt.savefig("accuracy_chart.png")
            plt.close()
            mlflow.log_artifact("accuracy_chart.png")

            # 5. Логируем саму модель как артефакт MLflow            
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                serialization_format="cloudpickle"
            )
            logging.info(f"✅ Запуск {run_name} завершен | Top-1: {acc_top1:.4f} | Top-2: {acc_top2:.4f}")

            if acc_top2 > best_acc:
                best_acc = acc_top2
                best_run_id = run.info.run_id

    # 6. Регистрируем лучшую модель в реестре и переводим в Staging
    if best_run_id:
        model_uri = f"runs:/{best_run_id}/model"
        model_name = "FilmGenreClassifier"
        reg_model = mlflow.register_model(model_uri, model_name)
        
        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=reg_model.version,
            stage="Staging"
        )
        logging.info(f"🏆 Лучшая модель (v{reg_model.version}, Top-2={best_acc:.4f}) зарегистрирована и переведена в STAGING!")

if __name__ == "__main__":
    run_mlflow_experiments()
