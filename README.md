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

- action mse loss
```
python train_light.py --env hopper --dataset medium-replay --model ofa
```
- action bce loss
```
python train_light.py --env hopper --dataset medium-replay --model ofa --criterion label_smoothed_cross_entropy
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
