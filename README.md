# embodiedMM:
Embodied Multi-Modal Artificial Intelligence

## Training

### dt
```
python train_light.py --env hopper --dataset medium-replay --model dt
```

### bc (to be tested)
```
python train_light.py --env hopper --dataset medium-replay --model bc
```

### ofa

- action mse loss (not work)
```
python train_light.py --env hopper --dataset medium-replay --model ofa
```
- action bce loss
```
sh run_scripts/rl/rl_ofa_try.sh
```

### pretrained task + ofa

- GPU
```
sh run_scripts/pretraining/pretrain_ofa_base.sh
```
- CPU
```
sh run_scripts/pretraining/pretrain_ofa_trial.sh
```

### metamorph example
```
cd metamorph
python tools/train_ppo.py --cfg ./configs/ft.yaml
```
