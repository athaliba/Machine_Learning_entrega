"""Executa preparação, seleção por validação cruzada, treino e avaliação."""

from loguru import logger

from module_olist.config import INTERIM_DATA_DIR, MODELS_DIR, RAW_DATA_DIR, REPORTS_DIR, FIGURES_DIR
from module_olist.dataset import create_dataset, load_dataset, save_dataset
from module_olist.features import create_features
from module_olist.modeling.evaluate import evaluate_model
from module_olist.modeling.cross_validation import cross_validate_models
from module_olist.modeling.train import train_model
from module_olist.split import split_data


def main():
    logger.info("Preparando a base de pedidos...")
    orders, items, customers = load_dataset(
        RAW_DATA_DIR / "olist_orders_dataset.csv",
        RAW_DATA_DIR / "olist_order_items_dataset.csv",
        RAW_DATA_DIR / "olist_customers_dataset.csv",
    )
    data = create_features(create_dataset(orders, items, customers))
    save_dataset(data, INTERIM_DATA_DIR / "orders_dataset_refined.csv")
    X_train, X_test, y_train, y_test = split_data(data)
    logger.info(f"Treino: {len(X_train):,} | Teste: {len(X_test):,}")

    # A escolha recebe apenas treino; o teste não participa desta busca.
    model_name, threshold = cross_validate_models(X_train, y_train)
    model = train_model(
        model_name=model_name, threshold=threshold,
        X_train=X_train, y_train=y_train,
        model_path=MODELS_DIR / "best_model.joblib",
        metadata_path=MODELS_DIR / "metadata_cv.json",
    )
    evaluate_model(
        model=model, model_name=model_name, threshold=threshold,
        X_test=X_test, y_test=y_test, X_train=X_train, y_train=y_train,
        metrics_path=REPORTS_DIR / "cv_test_metrics.json",
        figure_path=FIGURES_DIR / "cv_confusion_matrix.png",
    )
    logger.warning(
        "O teste já foi avaliado na Aula 07; esta execução não representa um teste inédito. "
        "Não use seus resultados para retunar modelo ou limiar."
    )
    logger.success("Validação cruzada, treinamento e avaliação concluídos.")


if __name__ == "__main__":
    main()
