import torch
import cv2
import json
import os
import yaml
import sys
import argparse
import os
import string
import tqdm
from IPython import embed
from easydict import EasyDict
from utils.util import str_filt

import numpy as np
from model import crnn
from utils import util, ssim_psnr, utils_moran, utils_crnn


DEFAULT_DEVICE = "cuda:0"
converter_crnn = utils_crnn.strLabelConverter(string.digits + string.ascii_lowercase)


def CRNN_init(model_path):
  model = crnn.CRNN(32, 1, 37, 256)
  model = model.to(DEFAULT_DEVICE)
  print('loading pretrained crnn model from %s' % model_path)
  model.load_state_dict(torch.load(model_path))
  return model


def parse_crnn_data(imgs_input):
  imgs_input = torch.nn.functional.interpolate(imgs_input, (32, 100), mode='bicubic')
  R = imgs_input[:, 0:1, :, :]
  G = imgs_input[:, 1:2, :, :]
  B = imgs_input[:, 2:3, :, :]
  tensor = 0.299 * R + 0.587 * G + 0.114 * B
  return tensor


def get_transform_matrix(
    left: float,
    top: float,
    width: float,
    height: float,
    angle: float,
    target_width: float,
    target_height: float,
) -> np.ndarray:
  """Returns the affine transform matrix for the given parameters.

  Args:
    left: X-axis of the source box's left-top corner.
    top: Y-axis of the source box's left-top corner.
    width: Width of the source box.
    height: Height of the source box.
    angle: Rotation angle of the source box with respect to its left-top corner.
    target_width: Width of the target box.
    target_height: Height of the target box.

  Returns:
    affine_matrix: A 2x3 np.ndarray object denoting the transformation matrix to
      convert the rotated source box to the axis-aligned target box.
  """
  src_pts = np.array(
      [
          [left, top],
          _rotate([left, top], [left + width, top], np.pi * angle / 180),
          _rotate([left, top], [left, top + height], np.pi * angle / 180),
      ],
      dtype=np.float32,
  )
  dst_pts = np.array(
      [[0, 0], [target_width, 0], [0, target_height]], dtype=np.float32
  )
  return cv2.getAffineTransform(src_pts, dst_pts)


def _rotate(pt0, pt1, angle) -> list[float]:
  """Rotates the point pt1 with respect to the origin pt0.

  Args:
    pt0: (x0, y0), coordinates of the first point, also the origin.
    pt1: (x1, y1), coordinates of the second point.
    angle: A float denoting the rotation angle in rad.

  Returns:
    pt: Rotation result (x', y') of pt1 with respect to the origin pt0.
  """
  x0, y0 = pt0
  x1, y1 = pt1
  cosv, sinv = np.cos(angle), np.sin(angle)
  x1, y1 = x1 - x0, y1 - y0
  x1, y1 = x1 * cosv - y1 * sinv, x1 * sinv + y1 * cosv
  return [x1 + x0, y1 + y0]


def gather_outputs(crnn, image, annot):
  """Computes the per-image metrics."""
  outputs = []
  for box_index, per_box_annot in enumerate(annot['annotations']):
    rotated_box = per_box_annot['box']
    text = per_box_annot['text']

    target_width, target_height = rotated_box['width'], rotated_box['height']
    affine_matrix = get_transform_matrix(rotated_box['left'],
                                         rotated_box['top'],
                                         rotated_box['width'],
                                         rotated_box['height'],
                                         rotated_box['angle'],
                                         target_width,
                                         target_height)
    cropped = cv2.warpAffine(image, affine_matrix, (target_width, target_height))
    cropped = cropped[np.newaxis, ...]  # Add batch dimension.
    cropped = cropped.astype(np.float32) / 255.0
    cropped = torch.from_numpy(cropped).permute((0, 3, 1, 2))  # torch (b, c, h, w)
    cropped = cropped.to(DEFAULT_DEVICE)

    # Run OCR recognition.
    crnn_input = parse_crnn_data(cropped)
    crnn_output = crnn(crnn_input)
    _, preds = crnn_output.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    preds_size = torch.IntTensor([crnn_output.size(0)])
    pred_str_sr = converter_crnn.decode(preds.data, preds_size.data, raw=False)

    outputs.append({'pred': pred_str_sr, 'gt': text, 'size': target_height})
  return outputs


def bucketize(size):
  if size < 32: return 'small'
  if size < 64: return 'medium'
  return 'large'

def summarize(outputs):
  metrics = {}
  for output in outputs:
    gt, pred, size = output['gt'], output['pred'], output['size']
    bucket = bucketize(size)

    summary = metrics.setdefault(bucket, {'n_examples': 0, 'n_correct': 0})
    summary['n_examples'] += 1
    if str_filt(gt, 'lower') == pred:
      summary['n_correct'] += 1

  for bucket in ['tiny', 'small', 'medium', 'large']:
    if bucket in metrics:
      n_correct = metrics[bucket]['n_correct']
      n_examples = metrics[bucket]['n_examples']
      accuracy = n_correct / max(1, n_examples)
      print('%s: %.2lf (%d/%d)' % (bucket, accuracy * 100, n_correct, n_examples))


def main(args):
  output_json_path = args.image_dir.rstrip('/') + '.crnn_ocr.json'
  if not os.path.exists(output_json_path):
    crnn = CRNN_init(args.crnn_path)
    crnn.eval()

    outputs = []
    for file_name in tqdm.tqdm(sorted(os.listdir(args.annotation_dir))):
      file_id = file_name.split('.')[0]
      image_path = os.path.join(args.image_dir, f'{file_id}.png')
      annot_path = os.path.join(args.annotation_dir, f'{file_id}.json')
      image = cv2.imread(image_path)
      assert image.shape == (1024, 1024, 3), image.shape
      with open(annot_path, 'r') as f:
        annot = json.load(f)
      image = image[:, :, ::-1]  # bgr to rgb
      outputs.extend(gather_outputs(crnn, image, annot))
    with open(output_json_path, 'w') as f:
      f.write(json.dumps(outputs, indent=2))
  else:
    with open(output_json_path, 'r') as f:
      outputs = json.load(f)
  summarize(outputs)
  print('Done')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('--crnn_path', type=str, default=None)
  parser.add_argument('--annotation_dir', type=str, default=None)
  parser.add_argument('--image_dir', type=str, default=None)

  main(parser.parse_args())
