# baseline2
baseline.build_model
```
epochs=40, training_ratio=[3]*30 + [2]*10
```

# SN2
SN.build_model
loss_weights = [1, 0]
```
epochs=100, training_ratio=[3] * 40 + [2] * 40 + [1] * 20
```

# SN
SN.build_model
loss_weights = [1, 1e-6]
```
epochs=200, training_ratio=[3] * 80 + [2] * 80 + [1] * 40
```