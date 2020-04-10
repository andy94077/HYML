# test.h5 test2.h5
	test.h5 reads images by img_name = os.listdir()
	test2.h5 reads images by img_name = sorted(os.listdir())

	optimizer = Adam(1.5e-3)

    model = Sequential([
        Conv2D(64, 3, padding='same', activation='relu', input_shape=input_shape),
        Conv2D(64, 3, padding='same', activation='relu'),
        Conv2D(64, 3, strides=2, padding='same', activation='relu'),
        BatchNormalization(),

        Conv2D(128, 3, padding='same', activation='relu'),
        Conv2D(128, 3, strides=2, padding='same', activation='relu'),
        BatchNormalization(),

        Conv2D(256, 3, strides=2, padding='same', activation='relu'),
        BatchNormalization(),

        Conv2D(512, 3, strides=2, padding='same', activation='relu'),
        BatchNormalization(),

        Conv2D(512, 3, strides=2, padding='same', activation='relu'),
        BatchNormalization(),

        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.3),
        Dense(1024, activation='relu'),
        Dropout(0.3),
        Dense(output_dim, activation='softmax')
        ])

# model2.h5
	
    model = Sequential([
        Conv2D(64, 3, padding='same', activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(64, 3, padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, 3, strides=2, padding='same', activation='relu'),
        BatchNormalization(),

        Conv2D(128, 3, strides=2, padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, 3, padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, 3, padding='same', activation='relu'),
        BatchNormalization(),

        Conv2D(256, 3, strides=2, padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(256, 3, padding='same', activation='relu'),
        BatchNormalization(),

        Conv2D(512, 3, strides=2, padding='same', activation='relu'),
        BatchNormalization(),

        Conv2D(512, 3, strides=2, padding='same', activation='relu'),
        BatchNormalization(),

        GlobalAveragePooling2D(),
        Dense(1024, activation='relu'),
        Dense(output_dim, activation='softmax')
        ])
