## best.py
### best1.h5
	val_acc
	lr=1e-4, batch_size=64
	ii = Input((input_dim,), name='input')
	out = Dense(1, activation='sigmoid', name='output')(ii)

	Training score: [0.26705691237356977, 0.8847863038288173]
	Validaiton score: [0.28459677754184615, 0.8781566818408703]
	Testing score: 0.88979

### best32-1.h5
	val_loss
    ii = Input((input_dim,), name='input')
    x = Dense(32, activation='sigmoid')(ii)
    out = Dense(1, activation='sigmoid', name='output')(x)

	Training score: [0.2473020211290296, 0.8941041551937289]
	Validaiton score: [0.27951628723726846, 0.8788940090298103]
	Testing score: 0.88892

