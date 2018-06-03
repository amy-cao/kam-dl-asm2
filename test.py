from keras.models import load_model
from keras.utils import to_categorical

from image_utils import load_val_data


model = load_model('../saved_models/base_model_bz32_ep100_acc_valacc.h5')
X_test, y_test = load_val_data()
X_test = X_test.astype('float64')
X_test /= 255
y_test = to_categorical(y_test, num_classes=NUM_CLASSES)
print(model.evaluate(x=X_test, y=y_test, batch_size=BATCH_SIZE))
