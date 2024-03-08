import random

from data.image_generator import (ConformalWeldingImageGenerator,
                                  PolygonImageGenerator)

path = 'img/generated'

def main():
    h, w = 256, 256
    point_n = 500
    
    cw_gen = ConformalWeldingImageGenerator(h, w, point_n)
    polygon_gen = PolygonImageGenerator(h, w)

    for k in range(10):
        k = (k+1)*2
        for scale in range(10):
            scale = (scale + 1) / 10 * k
            img = cw_gen.generate_image(k, scale)
            cw_gen.save_image(img, f'{path}/cw_{k}_{scale}.png')
            
            for d in range(random.randint(0, 5)):
                cw_distort_rate = 0.03
                dimg = cw_gen.distort_image(img, cw_distort_rate)
                cw_gen.save_image(dimg, f'{path}/cw_{k}_{scale}_{d}.png')

    for n in range(3, 20):
        img = polygon_gen.generate_image(n)
        polygon_gen.save_image(img, f'{path}/polygon_{n}.png')
        for d in range(random.randint(0, 5)):
            polygon_distort_rate = random.choice([0.0025, 0.005, 0.0075, 0.01])
            dimg = polygon_gen.distort_image(img, polygon_distort_rate)
            polygon_gen.save_image(dimg, f'{path}/polygon_{n}_{d}.png')

