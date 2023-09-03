class Callback:
    def pre_sampling(self, elapsed_steps: int) -> None:
        pass

    def post_sampling(self, elapsed_steps: int) -> None:
        pass

    def pre_evaluation(self, elapsed_steps: int) -> None:
        pass

    def post_evaluation(self, elapsed_steps: int) -> None:
        pass

    def pre_optimization(self, elapsed_steps: int) -> None:
        pass

    def post_optimization(self, elapsed_steps: int) -> None:
        pass
