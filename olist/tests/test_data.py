from module_olist import config


def test_caminhos_partem_da_raiz_do_projeto():
    """Nenhum caminho deve depender da máquina de quem roda o projeto."""
    assert config.DATA_DIR.parent == config.PROJ_ROOT
    assert config.MODELS_DIR.parent == config.PROJ_ROOT
    assert config.FIGURES_DIR.parent == config.REPORTS_DIR
    assert (config.PROJ_ROOT / "pyproject.toml").exists()


def test_pastas_versionadas_existem():
    assert config.MODELS_DIR.exists()
    assert config.REPORTS_DIR.exists()
