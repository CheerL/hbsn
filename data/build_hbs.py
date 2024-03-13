#! ~/.pyenv/versions/matlab/bin/python

import os

import click
import matlab.engine
from genericpath import exists
from loguru import logger

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIMEOUT = 60

try:
    from utils.timeout import timeout
except ModuleNotFoundError:
    import sys
    sys.path.append(ROOT_DIR)
    from utils.timeout import timeout



def init_setting(eng):
    matlab_dir = f'{ROOT_DIR}/../hbs_seg'
    init_script = f'''
    addpath('{matlab_dir}');
    addpath('{matlab_dir}/dependencies');
    addpath('{matlab_dir}/dependencies/im2mesh');
    addpath('{matlab_dir}/dependencies/mfile');
    addpath('{matlab_dir}/dependencies/map/map');
    '''
    eng.eval(init_script, nargout=0)

@timeout(TIMEOUT)
def generate_hbs(eng, image_path, mat_path):
    hbs_script = f'''
    hbs = compute_hbs_from_image('{image_path}');
    save('{mat_path}', 'hbs');
    '''
    logger.info(f'Load image {image_path}')
    eng.eval(hbs_script, nargout=0)
    logger.info(f'Generated HBS to {mat_path}')


def start_eng(shared_name='hbs_engine'):
    eng = matlab.engine.start_matlab(f'-nojvm -nodisplay -nosplash -nodesktop -r "matlab.engine.shareEngine(\'{shared_name}\')"')
    init_setting(eng)
    logger.info('Started matlab eng')
    return eng

def restart_eng(eng, shared_name='hbs_engine'):
    logger.warning('Restarting matlab eng')
    eng.quit()
    eng = start_eng(shared_name)
    return eng


@click.command()
@click.option("--image_dir", default=f'{ROOT_DIR}/img/generated', help="Directory of images to generate HBS")
@click.option("--eng_name", default='hbs_engine', help="Shared name of the matlab engine")
def main(image_dir, eng_name):
    eng = start_eng(eng_name)

    for file in sorted(os.listdir(image_dir)):
        try:
            if file.endswith('.png'):
                image_path = os.path.join(image_dir, file)
                mat_path = image_path.replace('.png', '.mat')
                if exists(mat_path):
                    logger.info(f'HBS {mat_path} already exists')
                    continue

                try:
                    generate_hbs(eng, image_path, mat_path)
                except TimeoutError:
                    eng = restart_eng(eng, eng_name)
                    os.remove(image_path)
                    logger.error(f"Timeout, failed to generate and removed image {image_path}")
        except Exception as e:
            logger.error(e)
            continue

    eng.quit()

if __name__ == "__main__":
    main()
