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
@click.option("--no_cw", default=False, is_flags=True, help="Stop using CW")
@click.option("--no_poly", default=False, is_flags=True, help="Stop using poly")
@click.option("--cw_min_size", default=1, help="Min number of k in CW")
@click.option("--cw_max_size", default=15, help="Max number of k in CW")
@click.option("--cw_repeat_time", default=5, help="Repeat times of CW")
@click.option("--cw_noise_time", default=5, help="Noise times of CW")
@click.option("--poly_min_size", default=3, help="Min angle in poly")
@click.option("--poly_max_size", default=15, help="Max angle in poly")
@click.option("--poly_repeat_time", default=5, help="Repeat times of poly")
@click.option("--poly_noise_time", default=5, help="Noise times of poly")
def main(prefix, image_dir, h, w, point_n, no_cw, no_poly, 
         cw_min_size, cw_max_size, cw_repeat_time, cw_noise_time,
         poly_min_size, poly_max_size, poly_repeat_time, poly_noise_time):
    if not exists(image_dir):
        os.makedirs(image_dir)

    cw_gen = ConformalWeldingImageGenerator(h, w, point_n)
    poly_gen = PolygonImageGenerator(h, w)

    if not no_cw:
        generate_by_cw(prefix, image_dir, cw_gen, cw_min_size, cw_max_size, cw_repeat_time, cw_noise_time)

    if not no_poly:
        generate_by_poly(prefix, image_dir, poly_gen, poly_min_size, poly_max_size, poly_repeat_time, poly_noise_time)
                    
def generate_by_cw(prefix, image_dir, cw_gen, cw_min_size, cw_max_size, cw_repeat_time, cw_noise_time):
    cw_distort_rate_list = [0.015, 0.02, 0.025, 0.03, 0.035]
    
    for size in range(cw_min_size, cw_max_size):
        size = 2*size
        for scale in range(10):
            scale = (scale + 1) / 10 * size
            for r in range(cw_repeat_time):
                img = cw_gen.generate_image(size, scale)
                cw_gen.save_image(img, f'{image_dir}/{prefix}_cw_{size}_{scale}_{r}.png')
            
                for n in range(random.randint(0, cw_noise_time)):
                    cw_distort_rate = random.choice(cw_distort_rate_list)
                    dimg = cw_gen.distort_image(img, cw_distort_rate)
                    cw_gen.save_image(dimg, f'{image_dir}/{prefix}_cw_{size}_{scale}_{r}.{cw_distort_rate}_{n}.png')
                    
def generate_by_poly(prefix, image_dir, poly_gen, poly_min_size, poly_max_size, poly_repeat_time, poly_noise_time):
    poly_distort_rate_list = [0.005, 0.01, 0.015, 0.02, 0.025]

    for size in range(poly_min_size, poly_max_size):
        for r in range(poly_repeat_time):
            img = poly_gen.generate_image(size)
            poly_gen.save_image(img, f'{image_dir}/{prefix}_polygon_{size}_{r}.png')
            for n in range(random.randint(0, poly_noise_time)):
                poly_distort_rate = random.choice(poly_distort_rate_list)
                dimg = poly_gen.distort_image(img, poly_distort_rate)
                poly_gen.save_image(dimg, f'{image_dir}/{prefix}_polygon_{size}_{r}.{poly_distort_rate}_{n}.png')

if __name__ == '__main__':
    main()
