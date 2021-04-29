import torch


class BaseAgent:
    """
    The agent performs many functions, including: action-selection during
    sampling, returning policy-related values to use in training (e.g. action
    probabilities), storing recurrent state during sampling, managing model
    device, and performing model parameter communication between processes.
    The agent is both interfaces: sampler<-->neural network<-->algorithm.
    Typically, each algorithm and environment combination will require at
    least some of its own agent functionality.

    The base agent automatically carries out some of these roles.  It assumes
    there is one neural network model.  Agents using multiple models might
    need to extend certain funcionality to include those models, depending on
    how they are used.
    """
    def __init__(self, ModelCls=None, model_kwargs=None, initial_model_state_dict=None):
        """
        Arguments are saved but no model initialization occurs.

        Args:
            ModelCls: The model class to be used.
            model_kwargs (optional): Any keyword arguments to pass when instantiating the model.
            initial_model_state_dict (optional): Initial model parameter values.
        """

        if model_kwargs is None:
            model_kwargs = dict()

        self.ModelCls = ModelCls
        self.model_kwargs = model_kwargs

        self._model = None
        self._distribution = None
        self.device = None
        self._mode = None

    @property
    def model(self):
        return self._model

    @property
    def distribution(self):
        return self._distribution

    def initialize(self, env_spaces, share_memory=False):
        """
        Instantiates the neural net model(s) according to the environment
        interfaces.  

        Uses shared memory as needed--e.g. in CpuSampler, workers have a copy
        of the agent for action-selection.  The workers automatically hold
        up-to-date parameters in ``model``, because they exist in shared
        memory, constructed here before worker processes fork. Agents with
        additional model components (beyond ``self.model``) for
        action-selection should extend this method to share those, as well.

        Typically called in the sampler during startup.

        Args:
            env_spaces: passed to ``make_env_to_model_kwargs()``, typically namedtuple of 'observation' and 'action'.
            share_memory (bool): whether to use shared memory for model parameters.
        """
        self.env_model_kwargs = self.make_env_to_model_kwargs(env_spaces)
        self.create_model()
        self.device = torch.device("cpu")
        self.create_distribution()
        self.env_spaces = env_spaces
        self.share_memory = share_memory

    def make_env_to_model_kwargs(self, env_spaces):
        """Generate any keyword args to the model which depend on environment interfaces."""
        return {}

    def create_model(self):
        self.model = self.ModelCls(**self.env_model_kwargs, **self.model_kwargs)
        if self.initial_model_state_dict is not None:
            self.model.load_state_dict(self.initial_model_state_dict)

    def create_distribution(self):
        pass

    def to_device(self, cuda_idx=None):
        """Moves the model to the specified cuda device, if not ``None``.  If
        sharing memory, instantiates a new model to preserve the shared (CPU)
        model.  Agents with additional model components (beyond
        ``self.model``) for action-selection or for use during training should
        extend this method to move those to the device, as well.

        Typically called in the runner during startup.
        """
        if cuda_idx is None:
            return
        if self.shared_model is not None:
            self.model = self.ModelCls(**self.env_model_kwargs,
                **self.model_kwargs)
            self.model.load_state_dict(self.shared_model.state_dict())
        self.device = torch.device("cuda", index=cuda_idx)
        self.model.to(self.device)
        logger.log(f"Initialized agent model on device: {self.device}.")

    def evaluate(self, observation, prev_action, prev_reward, prev_rnn_state=None):
        """Returns values from model forward pass on training data (i.e. used
        in algorithm)."""
        raise NotImplementedError

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward, prev_rnn_state=None):
        """Returns selected actions for environment instances in sampler."""
        raise NotImplementedError

    def parameters(self):
        """Parameters to be optimized (overwrite in subclass if multiple models)."""
        return self.model.parameters()

    def state_dict(self):
        """Returns model parameters for saving."""
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        """Load model parameters, should expect format returned from ``state_dict()``."""
        self.model.load_state_dict(state_dict)

    def train_mode(self, itr):
        """Go into training mode (e.g. see PyTorch's ``Module.train()``)."""
        self.model.train()
        self._mode = "train"

    def sample_mode(self, itr):
        """Go into sampling mode."""
        self.model.eval()
        self._mode = "sample"

    def eval_mode(self, itr):
        """Go into evaluation mode.  Example use could be to adjust epsilon-greedy."""
        self.model.eval()
        self._mode = "eval"
