from module_reconhecimento_facial import config


def test_caminhos_partem_da_raiz_do_projeto():
    """Nenhum caminho deve depender da máquina de quem roda o projeto."""
    assert config.AUTHORIZED_DIR.parents[1] == config.PROJ_ROOT
    assert config.TEST_DIR.parents[1] == config.PROJ_ROOT
    assert (config.PROJ_ROOT / "pyproject.toml").exists()


def test_configuracao_do_reconhecimento():
    assert config.MODEL_NAME == "Facenet512"
    assert config.DISTANCE_METRIC == "cosine"
    assert ".jpg" in config.IMAGE_EXTENSIONS
