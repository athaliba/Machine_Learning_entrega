"""Executa a preparação, o treinamento e a avaliação da Aula 07."""

from loguru import logger

from module_olist.config import INTERIM_DATA_DIR, MODELS_DIR, RAW_DATA_DIR
from module_olist.dataset import create_dataset, load_dataset, save_dataset
from module_olist.features import create_features
from module_olist.modeling.evaluate import evaluate_model
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

    # Configuração inicial definida antes de olhar os resultados do teste.
    model_name = "Gradient Boosting"
    threshold = 0.50
    model = train_model(
        model_name=model_name, threshold=threshold,
        X_train=X_train, y_train=y_train,
        model_path=MODELS_DIR / "gradient_boosting.joblib",
        metadata_path=MODELS_DIR / "metadata.json",
    )
    evaluate_model(
        model=model, model_name=model_name, threshold=threshold,
        X_test=X_test, y_test=y_test, X_train=X_train, y_train=y_train,
    )
    logger.success("Treinamento e avaliação concluídos.")


if __name__ == "__main__":
    main()
