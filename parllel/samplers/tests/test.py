from parllel.samplers.tests.build import build_sampler

def test_single_batch():
    sampler = build_sampler(20, 4, recurrent=False)

    samples, completed_trajectories = sampler.collect_batch(elapsed_steps=0)

    print(samples)
    print(completed_trajectories)

if __name__ == "__main__":
    test_single_batch()