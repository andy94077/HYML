# lstm128.h5
	build_model2
    model = Sequential([
            Embedding(input_dim=embedding.shape[0], output_dim=embedding.shape[1], weights=[embedding]),
            Bidirectional(CuDNNGRU(128, return_sequences=True), merge_mode='concat'),
            Bidirectional(CuDNNGRU(128), merge_mode='concat'),
            Dense(1, activation='sigmoid')
        ])

# lstm128-3.h5 lstm128-3-2.h5
    model = Sequential([
            Embedding(input_dim=embedding.shape[0], output_dim=embedding.shape[1], weights=[embedding], trainable=False),
            Bidirectional(CuDNNGRU(128, return_sequences=True), merge_mode='concat'),
            Bidirectional(CuDNNGRU(128), merge_mode='concat'),
            Dropout(0.2),
            Dense(1, activation='sigmoid', kernel_regularizer=l2(1e-3))
        ])

# lstm128_semi.h5
    model = Sequential([
            Embedding(input_dim=embedding.shape[0], output_dim=embedding.shape[1], weights=[embedding], trainable=False),
            Bidirectional(CuDNNGRU(128, return_sequences=True), merge_mode='concat'),
            Bidirectional(CuDNNGRU(128), merge_mode='concat'),
            Dropout(0.2),
            Dense(1, activation='sigmoid', kernel_regularizer=l2(1e-3))
        ])

        threshold = 0.0
        for _ in range(3):
            model.load_weights(model_path)
            Y = model.predict(trainX_no_label, batch_size=512, verbose=1).ravel()
            trainX_aug = np.concatenate([trainX, trainX_no_label[(Y >= threshold) | (Y <= 1 - threshold)]], axis=0)
            trainY_aug = np.concatenate([trainY, np.round(Y[(Y >= threshold) | (Y <= 1 - threshold)])])

            reduce_lr = ReduceLROnPlateau('val_loss', 0.8, 0, verbose=1, min_lr=1e-5)
            model.fit(trainX_aug, trainY_aug, validation_data=(validX, validY), batch_size=512, epochs=5, callbacks=[checkpoint, reduce_lr])

# lstm128_semi-2.h5
    model = Sequential([
            Embedding(input_dim=embedding.shape[0], output_dim=embedding.shape[1], weights=[embedding], trainable=False),
            Bidirectional(CuDNNGRU(128, return_sequences=True), merge_mode='concat'),
            Bidirectional(CuDNNGRU(128), merge_mode='concat'),
            Dropout(0.2),
            Dense(1, activation='sigmoid', kernel_regularizer=l2(1e-3))
        ])

        threshold = 0.0
		model.load_weights(model_path)
		Y = model.predict(trainX_no_label, batch_size=512, verbose=1).ravel()
		trainX_aug = np.concatenate([trainX, trainX_no_label[(Y >= threshold) | (Y <= 1 - threshold)]], axis=0)
		trainY_aug = np.concatenate([trainY, np.round(Y[(Y >= threshold) | (Y <= 1 - threshold)])])

		reduce_lr = ReduceLROnPlateau('val_loss', 0.8, 0, verbose=1, min_lr=1e-5)
		model.fit(trainX_aug, trainY_aug, validation_data=(validX, validY), batch_size=512, epochs=5, callbacks=[reduce_lr])
		model.save_weights(model_path)

# lstm128-3_acc.h5
	main_acc.py
    model = Sequential([
            Embedding(input_dim=embedding.shape[0], output_dim=embedding.shape[1], weights=[embedding], trainable=False),
            Bidirectional(CuDNNGRU(128, return_sequences=True), merge_mode='concat'),
            Bidirectional(CuDNNGRU(128), merge_mode='concat'),
            Dropout(0.2),
            Dense(1, activation='sigmoid', kernel_regularizer=l2(5e-4))
        ])

