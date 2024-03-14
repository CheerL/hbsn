#! /home/extradisk/linchenran/.pyenv/versions/hbs_seg/bin/python
import os
import random

import click
from genericpath import exists

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    from data.image_generator import (ConformalWeldingImageGenerator,
                                      PolygonImageGenerator)
except ModuleNotFoundError:
    import sys
    sys.path.append(ROOT_DIR)
    from data.image_generator import (ConformalWeldingImageGenerator,
                                      PolygonImageGenerator)


@click.command()
@click.option("--prefix", default='No', help="Prefix of the generated images")
@click.option("--image_dir", default='img/generated', help="Directory of images to generate HBS")
@click.option("--h", default=256, help="Height of the generated images")
@click.option("--w", default=256, help="Width of the generated images")
@click.option("--point_n", default=500, help="Number of points in the generated images")
def main(prefix, image_dir, h, w, point_n):
    if not exists(image_dir):
        os.makedirs(image_dir)

    distort_max_n = 20
    
    cw_k_size = 25
    cw_distort_rate_list = [0.015, 0.02, 0.025, 0.03, 0.035]
    
    poly_n_size = 25
    poly_k_size = 50
    poly_distort_rate_list = [0.005, 0.01, 0.015, 0.02, 0.025]
    
    cw_gen = ConformalWeldingImageGenerator(h, w, point_n)
    poly_gen = PolygonImageGenerator(h, w)

    for k in range(cw_k_size):
        k = (k+1)*2
        for scale in range(10):
            scale = (scale + 1) / 10 * k
            img = cw_gen.generate_image(k, scale)
            cw_gen.save_image(img, f'{image_dir}/{prefix}_cw_{k}_{scale}.png')
            
            for d in range(random.randint(0, distort_max_n)):
                cw_distort_rate = random.choice(cw_distort_rate_list)
                dimg = cw_gen.distort_image(img, cw_distort_rate)
                cw_gen.save_image(dimg, f'{image_dir}/{prefix}_cw_{k}_{scale}_{cw_distort_rate}_{d}.png')

    for n in range(3, poly_n_size):
        for k in range(poly_k_size):
            img = poly_gen.generate_image(n)
            poly_gen.save_image(img, f'{image_dir}/{prefix}_polygon_{n}_{k}.png')
            for d in range(random.randint(0, distort_max_n)):
                poly_distort_rate = random.choice(poly_distort_rate_list)
                dimg = poly_gen.distort_image(img, poly_distort_rate)
                poly_gen.save_image(dimg, f'{image_dir}/{prefix}_polygon_{n}_{k}_{poly_distort_rate}_{d}.png')

if __name__ == '__main__':
    main()
