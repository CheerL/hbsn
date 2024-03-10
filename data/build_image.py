import os
import random

from genericpath import exists
from data.image_generator import (ConformalWeldingImageGenerator,
                                  PolygonImageGenerator)


def main():
    image_dir = 'img/generated'
    if not exists(image_dir):
        os.makedirs(image_dir)

    h, w = 256, 256
    point_n = 500
    
    cw_gen = ConformalWeldingImageGenerator(h, w, point_n)
    polygon_gen = PolygonImageGenerator(h, w)

    for k in range(10):
        k = (k+1)*2
        for scale in range(10):
            scale = (scale + 1) / 10 * k
            img = cw_gen.generate_image(k, scale)
            cw_gen.save_image(img, f'{image_dir}/cw_{k}_{scale}.png')
            
            for d in range(random.randint(0, 5)):
                cw_distort_rate = random.choice([0.02, 0.025, 0.03])
                dimg = cw_gen.distort_image(img, cw_distort_rate)
                cw_gen.save_image(dimg, f'{image_dir}/cw_{k}_{scale}_{d}.png')

    for n in range(3, 15):
        for k in range(20):
            img = polygon_gen.generate_image(n)
            polygon_gen.save_image(img, f'{image_dir}/polygon_{n}_{k}.png')
            for d in range(random.randint(0, 5)):
                polygon_distort_rate = random.choice([0.005, 0.01, 0.015])
                dimg = polygon_gen.distort_image(img, polygon_distort_rate)
                polygon_gen.save_image(dimg, f'{image_dir}/polygon_{n}_{k}_{d}.png')

if __name__ == '__main__':
    main()