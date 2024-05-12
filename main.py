import sys

from train.hbsn import main as hbsn_train
from train.maskrcnn import main as maskrcnn_train
from train.deeplab import main as deeplab_train
from train.unetpp import main as unetpp_train

if __name__ == '__main__':
    model_type = sys.argv[1]
    train_func_dict = {
        'hbsn': hbsn_train,
        'maskrcnn': maskrcnn_train,
        'deeplab': deeplab_train,
        'unetpp': unetpp_train,
    }
    main = train_func_dict.get(model_type, None)
    
    if main:
        sys.argv.pop(1)
        main()
    else:
        raise ValueError('Unknown model type')