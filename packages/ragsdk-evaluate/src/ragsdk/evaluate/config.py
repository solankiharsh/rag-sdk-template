from ragsdk.core.config import CoreConfig  # type: ignore
from ragsdk.core.utils._pyproject import get_config_instance  # type: ignore


class EvaluateConfig(CoreConfig):
    """
    Configuration for the ragsdk-evaluate package, loaded from downstream projects'
    pyproject.toml files.
    """


eval_config = get_config_instance(EvaluateConfig, subproject="evaluate")
