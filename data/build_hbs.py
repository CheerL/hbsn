import os

import matlab.engine

from utils.timeout import timeout


def init_setting(eng):
    matlab_dir = '/home/extradisk/linchenran/hbs_seg'
    init_script = f'''
    clear all;
    close all;
    cd('{matlab_dir}');
    addpath('{matlab_dir}/dependencies');
    addpath('{matlab_dir}/dependencies/im2mesh');
    addpath('{matlab_dir}/dependencies/mfile');
    addpath('{matlab_dir}/dependencies/map/map');
    '''
    eng.eval(init_script, nargout=0)

@timeout(60)
def generate_hbs(eng, image_path, timeout=30):
    hbs_script = f'''
    hbs = compute_hbs_from_image('{image_path}');
    hbs_dict.{image_path.split('/')[-1].replace('.', '_')} = hbs;
    '''
    print(f'Load image {image_path} and try to generate HBS...')
    eng.eval(hbs_script, nargout=0)
    print('Generated HBS successfully')


def save_hbs_dict(eng, mat_path):
    save_script = f'''
    save('{mat_path}', 'hbs_dict');
    '''
    eng.eval(save_script, nargout=0)

def main():
    image_dir = '/home/extradisk/linchenran/hbs_torch/img/generated'
    eng_shared_name = 'hbs_engine'
    eng = matlab.engine.start_matlab(f'-nojvm -nodisplay -nosplash -nodesktop -r "matlab.engine.shareEngine(\'{eng_shared_name}\')"')

    init_setting(eng)
    for file in os.listdir(image_dir):
        if file.endswith('.png'):
            image_path = os.path.join(image_dir, file)
            try:
                generate_hbs(eng, image_path)
            except TimeoutError:
                os.remove(image_path)
                print(f"Failed and remove image {image_path}")

    save_hbs_dict(eng, f'{image_dir}/hbs_dict.mat')
    eng.quit()
    
if __name__ == "__main__":
    main()