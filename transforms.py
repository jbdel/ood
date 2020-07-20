import torch
import numpy as np

class RepeatGrayscaleChannels(object):

  def __init__(self, num_channels):
    self.num_channels = num_channels

  def __call__(self, image_arr):
    if len(image_arr.shape) == 2:
      return np.repeat(image_arr[:,:,np.newaxis], self.num_channels, axis=2)

    return image_arr

class Clahe(object):

  def __call__(self, image_arr):
    return image_arr


class RandomSquareCrop(object):

  def __init__(self, size):
    assert isinstance(size, (int, tuple))
    if isinstance(size, int):
      self.size = (size, size)
    else:
      assert len(size) == 2
      assert size[0] == size[1]
      self.size = size

  def __call__(self, image_arr):
    h, w = image_arr.shape[:2]
    new_h, new_w = self.size

    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)

    return image_arr[top:top+new_h, left:left+new_w]


class RandomHorizontalFlip(object):

  def __call__(self, image_arr):
    if np.random.random() < 0.5:
      return np.flip(image_arr, axis=1).copy()

    return image_arr


class Transpose(object):

  def __call__(self, image_arr):
    return image_arr.transpose((2, 0, 1))


class ToTensor(object):

  def __call__(self, numpy_arr):
    return torch.from_numpy(numpy_arr)


class ToFloat(object):

  def __call__(self, x):
    return x.float()


class ToLong(object):

  def __call__(self, x):
    return x.long()
