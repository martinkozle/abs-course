from PIL import Image
import gym
import numpy as np
from deep_q_learning import DuelingDQN
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D
from tensorflow.keras.models import Model

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input



if __name__ == '__main__':
    env = gym.make('MsPacman-v0')

    img = Image.fromarray(state)
    img2 = img.convert('L')

    img3 = np.array(img2, dtype=np.float)
    img3 /= 255

    agent = DuelingDQN(...)

    layers = [Conv2D(32, (8, 8), activation='relu'),
              ...,
              Flatten(),
              Dense(512, activation='relu'),
              ...]

    agent.build_model(layers)

    base_model = VGG16(include_top=True, weights='imagenet', input_shape=(224, 224, 3))

    x = Dense(32)(base_model.layers[-1].output)

    ...

    model = Model(inputs=base_model.inputs, outputs=q)














