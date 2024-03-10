from genericpath import exists
import os
import matlab.engine

# from pygments import highlight
# from pygments.lexers import PythonLexer
# from pygments.formatters import ImageFormatter

def init_setting(eng):
    init_script = f'''
    clear all;
    close all;
    cd('/home/extradisk/linchenran/hbs_seg');
    addpath('/home/extradisk/linchenran/hbs_seg/dependencies');
    addpath('/home/extradisk/linchenran/hbs_seg/dependencies/im2mesh');
    addpath('/home/extradisk/linchenran/hbs_seg/dependencies/mfile');
    addpath('/home/extradisk/linchenran/hbs_seg/dependencies/map/map');
    '''
    eng.eval(init_script, nargout=0)

def generate_hbs(eng, image_path):
    hbs_script = f'''
    hbs = compute_hbs_from_image('{image_path}');
    hbs_dict.{image_path.split('/')[-1].replace('.', '_')} = hbs;
    '''
    # print(hbs_script)
    eng.eval(hbs_script, nargout=0)

def save_hbs_dict(eng):
    save_script = '''
    save('hbs.mat', 'hbs_dict');
    '''
    eng.eval(save_script, nargout=0)

if __name__ == "__main__":
    path_dir = '/home/extradisk/linchenran/hbs_torch/img/generated'
    eng = matlab.engine.start_matlab('-nojvm -nodisplay -nosplash -nodesktop')
    
    if not exists(path_dir):
        os.mkdir(path_dir)

    init_setting(eng)
    for file in os.listdir(path_dir):
        if file.endswith('.png'):
            image_path = os.path.join(path_dir, file)
            generate_hbs(eng, image_path)
            print(image_path)
            
    save_hbs_dict(eng)
    eng.quit()
