import marimo

__generated_with = "0.9.14"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        r"""
        # Image classification from scratch

        **Author:** [fchollet](https://twitter.com/fchollet)<br>
        **Date created:** 2020/04/27<br>
        **Last modified:** 2023/11/09<br>
        **Description:** Training an image classifier from scratch on the Kaggle Cats vs Dogs dataset.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Introduction

        This example shows how to do image classification from scratch, starting from JPEG
        image files on disk, without leveraging pre-trained weights or a pre-made Keras
        Application model. We demonstrate the workflow on the Kaggle Cats vs Dogs binary
        classification dataset.

        We use the `image_dataset_from_directory` utility to generate the datasets, and
        we use Keras image preprocessing layers for image standardization and data augmentation.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Setup
        """
    )
    return


@app.cell
def __():
    import os
    import numpy as np
    import keras
    from keras import layers
    from tensorflow import data as tf_data
    import matplotlib.pyplot as plt
    return keras, layers, np, os, plt, tf_data


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Load the data: the Cats vs Dogs dataset

        ### Raw data download

        First, let's download the 786M ZIP archive of the raw data:
        """
    )
    return


app._unparsable_cell(
    r"""
    !curl -O https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip
    """,
    name="__"
)


app._unparsable_cell(
    r"""
    !unzip -q kagglecatsanddogs_5340.zip
    !ls
    """,
    name="__"
)


@app.cell
def __(mo):
    mo.md(
        r"""
        Now we have a `PetImages` folder which contain two subfolders, `Cat` and `Dog`. Each
        subfolder contains image files for each category.
        """
    )
    return


app._unparsable_cell(
    r"""
    !ls PetImages
    """,
    name="__"
)


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Filter out corrupted images

        When working with lots of real-world image data, corrupted images are a common
        occurence. Let's filter out badly-encoded images that do not feature the string "JFIF"
        in their header.
        """
    )
    return


@app.cell
def __(os):
    num_skipped = 0
    for folder_name in ('Cat', 'Dog'):
        folder_path = os.path.join('PetImages', folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                fobj = open(fpath, 'rb')
                is_jfif = b'JFIF' in fobj.peek(10)
            finally:
                fobj.close()
            if not is_jfif:
                num_skipped = num_skipped + 1
                os.remove(fpath)
    print(f'Deleted {num_skipped} images.')
    return (
        fname,
        fobj,
        folder_name,
        folder_path,
        fpath,
        is_jfif,
        num_skipped,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Generate a `Dataset`
        """
    )
    return


@app.cell
def __(keras):
    image_size = (180, 180)
    batch_size = 128

    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        "PetImages",
        validation_split=0.2,
        subset="both",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )
    return batch_size, image_size, train_ds, val_ds


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Visualize the data

        Here are the first 9 images in the training dataset.
        """
    )
    return


@app.cell
def __(np, plt, train_ds):
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(np.array(images[i]).astype("uint8"))
            plt.title(int(labels[i]))
            plt.axis("off")
    return ax, i, images, labels


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Using image data augmentation

        When you don't have a large image dataset, it's a good practice to artificially
        introduce sample diversity by applying random yet realistic transformations to the
        training images, such as random horizontal flipping or small random rotations. This
        helps expose the model to different aspects of the training data while slowing down
        overfitting.
        """
    )
    return


@app.cell
def __(layers):
    data_augmentation_layers = [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]

    def data_augmentation(images):
        for layer in data_augmentation_layers:
            images = layer(images)
        return images
    return data_augmentation, data_augmentation_layers


@app.cell
def __(mo):
    mo.md(
        r"""
        Let's visualize what the augmented samples look like, by applying `data_augmentation`
        repeatedly to the first few images in the dataset:
        """
    )
    return


@app.cell
def __(data_augmentation, np, plt, train_ds):
    plt.figure(figsize=(10, 10))
    for images_1, _ in train_ds.take(1):
        for i_1 in range(9):
            augmented_images = data_augmentation(images_1)
            ax_1 = plt.subplot(3, 3, i_1 + 1)
            plt.imshow(np.array(augmented_images[0]).astype('uint8'))
            plt.axis('off')
    return augmented_images, ax_1, i_1, images_1


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Standardizing the data

        Our image are already in a standard size (180x180), as they are being yielded as
        contiguous `float32` batches by our dataset. However, their RGB channel values are in
        the `[0, 255]` range. This is not ideal for a neural network;
        in general you should seek to make your input values small. Here, we will
        standardize values to be in the `[0, 1]` by using a `Rescaling` layer at the start of
        our model.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Two options to preprocess the data

        There are two ways you could be using the `data_augmentation` preprocessor:

        **Option 1: Make it part of the model**, like this:

        ```python
        inputs = keras.Input(shape=input_shape)
        x = data_augmentation(inputs)
        x = layers.Rescaling(1./255)(x)
        ...  # Rest of the model
        ```

        With this option, your data augmentation will happen *on device*, synchronously
        with the rest of the model execution, meaning that it will benefit from GPU
        acceleration.

        Note that data augmentation is inactive at test time, so the input samples will only be
        augmented during `fit()`, not when calling `evaluate()` or `predict()`.

        If you're training on GPU, this may be a good option.

        **Option 2: apply it to the dataset**, so as to obtain a dataset that yields batches of
        augmented images, like this:

        ```python
        augmented_train_ds = train_ds.map(
            lambda x, y: (data_augmentation(x, training=True), y))
        ```

        With this option, your data augmentation will happen **on CPU**, asynchronously, and will
        be buffered before going into the model.

        If you're training on CPU, this is the better option, since it makes data augmentation
        asynchronous and non-blocking.

        In our case, we'll go with the second option. If you're not sure
        which one to pick, this second option (asynchronous preprocessing) is always a solid choice.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Configure the dataset for performance

        Let's apply data augmentation to our training dataset,
        and let's make sure to use buffered prefetching so we can yield data from disk without
        having I/O becoming blocking:
        """
    )
    return


@app.cell
def __(data_augmentation, tf_data, train_ds, val_ds):
    train_ds_1 = train_ds.map(lambda img, label: (data_augmentation(img), label), num_parallel_calls=tf_data.AUTOTUNE)
    train_ds_1 = train_ds_1.prefetch(tf_data.AUTOTUNE)
    val_ds_1 = val_ds.prefetch(tf_data.AUTOTUNE)
    return train_ds_1, val_ds_1


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Build a model

        We'll build a small version of the Xception network. We haven't particularly tried to
        optimize the architecture; if you want to do a systematic search for the best model
        configuration, consider using
        [KerasTuner](https://github.com/keras-team/keras-tuner).

        Note that:

        - We start the model with the `data_augmentation` preprocessor, followed by a
         `Rescaling` layer.
        - We include a `Dropout` layer before the final classification layer.
        """
    )
    return


@app.cell
def __(image_size, keras, layers):
    def make_model(input_shape, num_classes):
        inputs = keras.Input(shape=input_shape)

        # Entry block
        x = layers.Rescaling(1.0 / 255)(inputs)
        x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        for size in [256, 512, 728]:
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(size, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(size, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.Conv2D(size, 1, strides=2, padding="same")(
                previous_block_activation
            )
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        x = layers.SeparableConv2D(1024, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.GlobalAveragePooling2D()(x)
        if num_classes == 2:
            units = 1
        else:
            units = num_classes

        x = layers.Dropout(0.25)(x)
        # We specify activation=None so as to return logits
        outputs = layers.Dense(units, activation=None)(x)
        return keras.Model(inputs, outputs)


    model = make_model(input_shape=image_size + (3,), num_classes=2)
    keras.utils.plot_model(model, show_shapes=True)
    return make_model, model


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Train the model
        """
    )
    return


@app.cell
def __(keras, model, train_ds_1, val_ds_1):
    epochs = 10
    callbacks = [keras.callbacks.ModelCheckpoint('save_at_{epoch}.keras')]
    model.compile(optimizer=keras.optimizers.Adam(0.0003), loss=keras.losses.BinaryCrossentropy(from_logits=True), metrics=[keras.metrics.BinaryAccuracy(name='acc')])
    model.fit(train_ds_1, epochs=epochs, callbacks=callbacks, validation_data=val_ds_1)
    return callbacks, epochs


@app.cell
def __(mo):
    mo.md(
        r"""
        We get to >90% validation accuracy after training for 25 epochs on the full dataset
        (in practice, you can train for 50+ epochs before validation performance starts degrading).
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Run inference on new data

        Note that data augmentation and dropout are inactive at inference time.
        """
    )
    return


@app.cell
def __(image_size, keras, model, plt):
    img = keras.utils.load_img("PetImages/Cat/10198.jpg", target_size=image_size)
    plt.imshow(img)

    img_array = keras.utils.img_to_array(img)
    img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    score = float(keras.ops.sigmoid(predictions[0][0]))
    print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")
    return img, img_array, predictions, score


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

