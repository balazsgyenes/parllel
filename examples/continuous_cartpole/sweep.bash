#!/bin/bash

for METHOD in fork forkserver
do
    for ITERATION in {1..10}
    do
        python examples/continuous_cartpole/train_ppo.py start_method=$METHOD
    done
done
