from easydict import EasyDict as edict
from pathlib import Path
import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms as trans

def get_config(training = True, args = None):
    conf = edict()
    conf.data_path = Path('data')
    conf.work_path = Path('work_space/')
    conf.model_path = conf.work_path/'models'
    conf.log_path = conf.work_path/'log'
    # conf.save_path = Path("work_space/save")
    # conf.save_path = Path("simclr/model.pth")
    conf.save_path = Path("MOCO/resnet50.pth")
    conf.input_size = [112,112]
    conf.embedding_size = 512
    conf.use_mobilfacenet = True
    conf.net_depth = 50
    conf.drop_ratio = 0.6
    conf.arch = args.arch
    conf.device = torch.device('cuda:{}'.format(args.gpu_id)) if args and args.gpu_id else torch.device('cpu')
    # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf.test_transform = trans.Compose([
                    trans.ToTensor(),
                    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
    conf.data_mode = 'emore'
    conf.vgg_folder = conf.data_path/'faces_vgg_112x112'
    conf.ms1m_folder = conf.data_path/'faces_ms1m_112x112'
    conf.emore_folder = conf.data_path/'faces_emore'
    conf.base_folder = conf.data_path/'facebank'/'distinct'/'raw_112'
    # conf.batch_size = 100 # irse net depth 50 
    conf.batch_size = 20 # mobilefacenet 200
#--------------------Training Config ------------------------    
    if training:        
        conf.log_path = conf.work_path/'log'
        # conf.save_path = Path("work_space/save")
        conf.save_path = Path("simclr/model.pth")
        # conf.weight_decay = 5e-4
        conf.lr = 1e-4 # originally 1e-3
        conf.milestones = [12,15,18]
        conf.momentum = 0.9
        conf.pin_memory = True
        conf.num_workers = 2 # 4 when batchsize is 200
        # conf.num_workers = 3 # when batchsize is 100
        conf.ce_loss = CrossEntropyLoss()    
#--------------------Inference Config ------------------------
    else:
        conf.facebank_path = conf.data_path/'facebank'
        conf.threshold = 1.5
        conf.face_limit = 1 #10 
        # when inference, at maximum detect 10 faces in one image, my laptop is slow
        conf.min_face_size = 30 
        # the larger this value, the faster deduction, comes with tradeoff in small faces
    return conf