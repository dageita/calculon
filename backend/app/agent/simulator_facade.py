"""Shared entry points for REST and LangChain tools."""

from app.core.calculate_repository import CalculateRepository
from app.models.calculator_input import Gpu, Model, Network, TrainningConfig, OptimalConfig


class SimulatorFacade:
    """Thin wrapper around CalculateRepository so HTTP and Agent share one code path."""

    def __init__(self) -> None:
        self._repo = CalculateRepository()

    def calculate(
        self,
        gpu: Gpu,
        network: Network,
        model: Model,
        trainning_config: TrainningConfig,
    ):
        return self._repo.calculate(gpu, network, model, trainning_config)

    def optimal(self, gpu: Gpu, network: Network, model: Model, optimal_config: OptimalConfig):
        return self._repo.optimal(gpu, network, model, optimal_config)
