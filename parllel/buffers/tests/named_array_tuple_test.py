import numpy as np

from parllel.buffers.named_tuple import NamedArrayTupleClass

if __name__ == "__main__":
    Cls = NamedArrayTupleClass("reward", ["gripper", "cauter"])

    tup = Cls(np.zeros(shape=(10,)), np.zeros(shape=(10,)))
    print(tup)

    step_reward = {
        "cauter": np.ones(shape=(1,)),
        "gripper": np.ones(shape=(1,)) * 2,
    }

    tup[3] = step_reward
    print(tup)

    step_reward = {
        "cauter": np.ones(shape=(1,))[0],
        "gripper": np.ones(shape=(1,))[0] * 2,
    }

    tup[7] = step_reward
    print(tup)
