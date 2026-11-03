"""Trains the name generator. Run with: python train.py"""

import keras

from slm.config import RACES, Config
from slm.dataset import build_dataset, load_names, split_by_name
from slm.model import build_model
from slm.tokenizer import PAD_ID, CharTokenizer


def main() -> None:
    config = Config()
    keras.utils.set_random_seed(config.seed)

    samples = load_names(config.names_path)
    train_samples, validation_samples = split_by_name(samples, config.validation_fraction, config.seed)
    print(f"{len(train_samples)} training names, {len(validation_samples)} validation names")

    # Fitted on all names, so a character that only occurs in the validation set
    # does not crash encoding. This reveals only which characters exist, not any names.
    tokenizer = CharTokenizer.fit([name for name, _ in samples])
    config.tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(config.tokenizer_path)

    train_data = build_dataset(train_samples, tokenizer, config.batch_size, shuffle=True)
    validation_data = build_dataset(validation_samples, tokenizer, config.batch_size)

    model = build_model(tokenizer.vocab_size, len(RACES), config)
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(config.learning_rate),
        # Measures how surprised the model was by each true next character. Padding
        # positions are ignored, so short names are not rewarded for predicting padding.
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=PAD_ID),
    )

    callbacks = [
        # Keeps the version of the model that did best on names it never trained on.
        keras.callbacks.ModelCheckpoint(config.model_path, monitor="val_loss", save_best_only=True),
        # Stops once validation loss has not improved for a while, which is the sign
        # that the model has begun memorising the training names.
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
        ),
    ]

    model.fit(
        train_data,
        validation_data=validation_data,
        epochs=config.max_epochs,
        callbacks=callbacks,
    )


if __name__ == "__main__":
    main()
