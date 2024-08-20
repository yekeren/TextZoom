import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf

_INPUT_IMAGE_COND_NAME = 'image_cond:0'
_OUTPUT_IMAGE_NAME = 'image'


def _resize_to_height(
    image: tf.Tensor,
    target_height: int,
    method: tf.image.ResizeMethod = tf.image.ResizeMethod.BILINEAR,
) -> tf.Tensor:
  """Resizes to match the `target_height` while keeping the aspect ratio.

  Args:
    image: A [height, width, channels] tf.uint8 tensor.
    target_height: Height of the image batch.
    method: Method for resizing.

  Returns:
    resized_image: A [target_height, target_width, channels] tf.uint8 tensor.
  """
  height, width, _ = tf.unstack(tf.shape(image))
  num_channels = image.shape[-1]

  # Resize while keeping the aspect ratio: `scale = target_height / height`.
  target_width = tf.maximum(1, tf.cast(width * target_height / height, tf.int32))
  resized_image = tf.image.resize(
      image, [target_height, target_width], method=method
  )
  resized_image.set_shape([None, None, num_channels])
  return tf.cast(resized_image, tf.uint8)



def _resize_and_left_pad(
    image: tf.Tensor, target_height: int, target_width: int
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Resizes to match `target_height` then left pads to match to `target_width`.

  This function first resize to match the target_height while keeping the
  aspect ratio. Then, it pads or trims the resized image to match to the
  target static shape [target_height, target_width] for batching.

  Args:
    image: A [height, width, channels] tf.uint8 tensor.
    target_height: Height of the image batch.
    target_width: Width of the image batch.

  Returns:
    resized_image: A [target_height, target_width, channels] tf.uint8 tensor.
    valid_height: Valid image height, should always be target_height.
    valid_width: Valid image width, valid_width <= target_width.
  """
  # Resize the image if necessary.
  height, num_channels = tf.shape(image)[0], image.shape[-1]
  image = tf.cond(
      height == target_height,
      true_fn=lambda: image,
      false_fn=lambda: _resize_to_height(image, target_height),
  )
  valid_height, valid_width, _ = tf.unstack(tf.shape(image))

  # Pad and trim to static shape [target_height, target_width].
  image = tf.cond(
      valid_width < target_width,
      lambda: tf.pad(
          image, paddings=[[0, 0], [0, target_width - valid_width], [0, 0]]
      ),
      lambda: image,
  )
  image = tf.cond(
      valid_width > target_width,
      lambda: image[:, :target_width, :],
      lambda: image,
  )
  image.set_shape([target_height, target_width, num_channels])
  return (
      tf.cast(image, tf.uint8),
      valid_height,
      tf.minimum(valid_width, target_width),
  )


class LumaText(object):
    def __init__(self, model_path: str, scale_factor=2, guidance_scale_img: float = 1.0):
        super().__init__()

        self._scale_factor = scale_factor
        self._guidance_scale = 0.0
        self._guidance_scale_img = guidance_scale_img

        self._imported = tf.saved_model.load(model_path)
        self._model_input_height = self._model_input_width = None
        input_tensors = self._imported.signatures['serving_default'].inputs
        for input_tensor in input_tensors:
            if input_tensor.name == _INPUT_IMAGE_COND_NAME:
                self._model_input_height = input_tensor.shape[1]
                self._model_input_width = input_tensor.shape[2]
                break
        if None in [self._model_input_height, self._model_input_width]:
            raise ValueError('Failed to find the input tensor in the TF graph.')
        print(f'Luma-text is initialized from {model_path}, expected input size={self._model_input_height}x{self._model_input_width}')

    def __call__(self, x):
        _, _, height, width = x.shape
        model_fn = self._imported.signatures['serving_default']

        def _process_single(image):
            model_input, valid_height, valid_width = _resize_and_left_pad(image,
                                                                          self._model_input_height,
                                                                          self._model_input_width)
            # Run TF engine, note that the TF engine requires a batch=1 first dimension.
            model_output = model_fn(guidance_scale=[self._guidance_scale],
                                    guidance_scale_img=[self._guidance_scale_img],
                                    image_cond=model_input[tf.newaxis, ...],
                                    tokens=[[0] * 128],
                                    valid_height=[valid_height],
                                    valid_width=[valid_width])[_OUTPUT_IMAGE_NAME]
            model_output = model_output[:, :, :valid_width, :]
            return model_output[0]


        def _process_batch(images):
            outputs = tf.map_fn(_process_single, images)
            return outputs


        # Adapt the input tensor.
        model_input = x.permute((0, 2, 3, 1))  # Torch2Tf ((b, c, h, w) -> (b, h, w, c)).
        model_input = model_input.cpu().numpy()  # CUDA to CPU.
        model_input = tf.cast(tf.clip_by_value(model_input, 0, 1) * 255, tf.uint8)  # Float to uint8.

        model_output = _process_batch(model_input)

        # Adapt the output tensor.
        model_output = tf.cast(model_output, tf.float32) / 255.0  # Uint8 to float.
        model_output = tf.transpose(model_output, (0, 3, 1, 2))  # Tf2Torch((b, h, w, c) -> (b, c, h, w))
        model_output = torch.from_numpy(model_output.numpy()).to('cuda')  # CPU to CUDA.
       
        target_h, target_w = height * self._scale_factor, width * self._scale_factor
        if (model_output.shape[2], model_output.shape[3]) != (target_h, target_w):
            output = F.interpolate(model_output, size=(target_h, target_w), mode='bicubic', align_corners=True)
        else:
            output = model_output
        return output
