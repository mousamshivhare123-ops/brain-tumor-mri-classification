"""Train and compare a custom CNN with MobileNetV2 transfer learning."""
import argparse, json
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42
CLASS_NAMES = ['Glioma', 'Meningioma', 'No_Tumor', 'Pituitary']


def load_datasets(data_dir):
    data_dir = Path(data_dir)
    common = dict(labels='inferred', label_mode='categorical', class_names=CLASS_NAMES,
                  image_size=IMG_SIZE, batch_size=BATCH_SIZE, seed=SEED)
    train = keras.utils.image_dataset_from_directory(data_dir/'train', shuffle=True, **common)
    valid = keras.utils.image_dataset_from_directory(data_dir/'valid', shuffle=False, **common)
    test = keras.utils.image_dataset_from_directory(data_dir/'test', shuffle=False, **common)
    autotune = tf.data.AUTOTUNE
    return train.prefetch(autotune), valid.prefetch(autotune), test.prefetch(autotune)


def callbacks(path):
    return [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(path, monitor='val_loss', save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-7),
    ]


def plot_history(history, name):
    Path('results').mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history.get('accuracy', []), label='Train')
    axes[0].plot(history.history.get('val_accuracy', []), label='Validation')
    axes[0].set_title('Accuracy'); axes[0].set_xlabel('Epoch'); axes[0].legend()
    axes[1].plot(history.history.get('loss', []), label='Train')
    axes[1].plot(history.history.get('val_loss', []), label='Validation')
    axes[1].set_title('Loss'); axes[1].set_xlabel('Epoch'); axes[1].legend()
    fig.tight_layout(); fig.savefig(f'results/{name}_history.png', dpi=160); plt.close(fig)
    Path(f'results/{name}_history.json').write_text(json.dumps(history.history, indent=2))


def build_custom():
    inp = keras.Input((*IMG_SIZE, 3))
    x = layers.Rescaling(1./255)(inp)
    x = layers.RandomFlip('horizontal')(x)
    x = layers.RandomRotation(0.05)(x)
    x = layers.RandomZoom(0.1)(x)
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.15)(x)
    x = layers.Conv2D(192, 3, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(len(CLASS_NAMES), activation='softmax')(x)
    model = keras.Model(inp, out, name='custom_cnn')
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_transfer():
    inp = keras.Input((*IMG_SIZE, 3))
    aug = keras.Sequential([
        layers.RandomFlip('horizontal'), layers.RandomRotation(0.05), layers.RandomZoom(0.1)
    ], name='augmentation')
    x = aug(inp)
    # Equivalent to MobileNetV2 preprocess_input for RGB pixels in [0, 255].
    x = layers.Rescaling(scale=1./127.5, offset=-1, name='mobilenet_preprocess')(x)
    base = keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(*IMG_SIZE, 3))
    base.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(len(CLASS_NAMES), activation='softmax')(x)
    model = keras.Model(inp, out, name='mobilenetv2_transfer')
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--fine-tune', action='store_true', help='Fine-tune top MobileNetV2 layers after initial training')
    args = ap.parse_args()
    Path('models').mkdir(exist_ok=True); Path('results').mkdir(exist_ok=True)
    train, valid, test = load_datasets(args.data)
    Path('models/class_names.json').write_text(json.dumps(CLASS_NAMES, indent=2))

    cnn = build_custom()
    hist_cnn = cnn.fit(train, validation_data=valid, epochs=args.epochs, callbacks=callbacks('models/custom_cnn.keras'))
    plot_history(hist_cnn, 'custom_cnn')
    cnn = keras.models.load_model('models/custom_cnn.keras')
    cnn.save('models/custom_cnn_final.keras')

    transfer, base = build_transfer()
    hist_transfer = transfer.fit(train, validation_data=valid, epochs=args.epochs, callbacks=callbacks('models/mobilenetv2.keras'))
    if args.fine_tune:
        base.trainable = True
        for layer in base.layers[:-30]: layer.trainable = False
        transfer.compile(optimizer=keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
        hist_ft = transfer.fit(train, validation_data=valid, epochs=max(5, args.epochs//2), callbacks=callbacks('models/mobilenetv2.keras'))
        # Record fine-tuning metrics separately to avoid conflating epochs.
        plot_history(hist_ft, 'mobilenetv2_finetune')
    else:
        plot_history(hist_transfer, 'mobilenetv2')
    transfer = keras.models.load_model('models/mobilenetv2.keras')
    transfer.save('models/mobilenetv2_final.keras')

    summary = {}
    for name, model in [('custom_cnn', cnn), ('mobilenetv2', transfer)]:
        loss, acc = model.evaluate(test, verbose=0)
        summary[name] = {'test_loss': float(loss), 'test_accuracy': float(acc)}
    Path('results/model_comparison.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print('Use validation results for model selection; test results are for final reporting only.')

if __name__ == '__main__':
    main()
