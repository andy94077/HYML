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

# lstm128-3_acc.h5
	main_acc.py
    model = Sequential([
            Embedding(input_dim=embedding.shape[0], output_dim=embedding.shape[1], weights=[embedding], trainable=False),
            Bidirectional(CuDNNGRU(128, return_sequences=True), merge_mode='concat'),
            Bidirectional(CuDNNGRU(128), merge_mode='concat'),
            Dropout(0.2),
            Dense(1, activation='sigmoid', kernel_regularizer=l2(5e-4))
        ])

