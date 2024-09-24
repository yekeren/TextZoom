import yaml
import sys
import argparse
import os
from IPython import embed
from easydict import EasyDict
from interfaces.super_resolution import TextSR


def main(config, args):
    Mission = TextSR(config, args)

    if args.test:
        Mission.test()
    elif args.demo:
        Mission.demo()
    else:
        Mission.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--arch', default='tsrn', choices=['tsrn', 'bicubic', 'srcnn', 'vdsr', 'srres', 'esrgan', 'rdn',
                                                           'edsr', 'lapsrn', 'luma-text'])
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--test_data_dir', type=str, default='../dataset/lmdb/str/TextZoom/test/medium/', help='')
    parser.add_argument('--batch_size', type=int, default=None, help='')
    parser.add_argument('--resume', type=str, default=None, help='')
    parser.add_argument('--vis_dir', type=str, default=None, help='')
    parser.add_argument('--rec', default='aster', choices=['aster', 'moran', 'crnn'])
    parser.add_argument('--STN', action='store_true', default=False, help='')
    parser.add_argument('--syn', action='store_true', default=False, help='use synthetic LR')
    parser.add_argument('--mixed', action='store_true', default=False, help='mix synthetic with real LR')
    parser.add_argument('--mask', action='store_true', default=False, help='')
    parser.add_argument('--gradient', action='store_true', default=False, help='')
    parser.add_argument('--hd_u', type=int, default=32, help='')
    parser.add_argument('--srb', type=int, default=5, help='')
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--demo_dir', type=str, default='./demo')

    # Below are the new arguments.
    parser.add_argument('--demo_output_dir', type=str, default='./results')
    parser.add_argument('--luma_model_path', type=str, default='', help='Path to the directory storing LUMA TF models')
    parser.add_argument('--guidance_scale_img', type=float, default=1.0)

    # Setting guidance_scale_txt > 0 enables text conditioning. The text source can be one from ['default', 'reference', 'crnn']:
    #     'default', means to use an empty string;
    #     'reference', means to use the ground truth text annotation;,
    #     'crnn', means to use the text extracted from the LQ image using the crnn ocr line recognizer.
    parser.add_argument('--guidance_scale_txt', type=float, default=0.0)
    parser.add_argument('--text_source', default='default', choices=['default', 'reference', 'aster', 'aster_fixed', 'moran', 'crnn'])
    args = parser.parse_args()
    config_path = os.path.join('config', 'super_resolution.yaml')
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    config = EasyDict(config)
    print(f'Recognizer: {args.rec}')
    print(f'Test data: {args.test_data_dir}')
    print(f'Guidance scale img: {args.guidance_scale_img}')
    main(config, args)
