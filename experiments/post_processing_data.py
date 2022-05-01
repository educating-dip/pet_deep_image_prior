from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
import numpy as np
import os
from omegaconf import OmegaConf
from tqdm import tqdm
from scipy.io import savemat
baseline_path_acc = glob.glob(os.getcwd() + "/baselines/*/*/*/events*")

if os.path.isfile(os.getcwd() + "/baseline_data.npy") == False:    
    data = []
    with tqdm(range(len(baseline_path_acc)), desc='Processing baseline data') as pbar:
        for i in pbar:
            event_acc = EventAccumulator(baseline_path_acc[i])
            event_acc.Reload()
            data_exp = {}
            conf = OmegaConf.load(os.path.dirname(os.path.dirname(baseline_path_acc[i]))+'/.hydra/config.yaml')['prior']
            data_exp['prior'] =conf['name']
            if conf['name'] != 'OSEM':
                data_exp['kappa'] = conf['kappa']
                data_exp['beta'] = conf['penalty_factor']
            CRCs = [(s.value) for s in event_acc.Scalars('CRC')]
            STDEVs = [(s.value) for s in event_acc.Scalars('STDEV')]
            recon = np.load(os.path.dirname(os.path.dirname(baseline_path_acc[i]))+'/recon.npy')
            profile = np.load(os.path.dirname(os.path.dirname(baseline_path_acc[i]))+'/profile.npy')
            data_exp['CRCs'] = CRCs
            data_exp['STD_DEVs'] = STDEVs
            data_exp['recon'] = recon
            data_exp['profile'] = profile
            data.append(data_exp)
    np.save('baseline_data',data)

main_path_acc = glob.glob(os.getcwd() + "/main/*/*/*/events*")

if os.path.isfile(os.getcwd() + "/main_data.npy") == False:
    data = []
    for i in range(len(main_path_acc)):
        data_exp = {}
        conf = OmegaConf.load(os.path.dirname(os.path.dirname(main_path_acc[i]))+'/.hydra/config.yaml')['prior']
        data_exp['prior'] =conf['name']
        if conf['name'] != 'OSEM':
            data_exp['kappa'] = conf['kappa']
            data_exp['beta'] = conf['penalty_factor']        
        recon = np.load(os.path.dirname(os.path.dirname(main_path_acc[i]))+'/recon.npy')
        profile = np.load(os.path.dirname(os.path.dirname(main_path_acc[i]))+'/profile.npy')
        crc = np.load(os.path.dirname(os.path.dirname(main_path_acc[i]))+'/crc.npy')
        std_dev = np.load(os.path.dirname(os.path.dirname(main_path_acc[i]))+'/std_dev.npy')
        data_exp['CRC'] = crc
        data_exp['STD_DEV'] = std_dev
        data_exp['recon'] = recon
        data_exp['profile'] = profile
        conf = OmegaConf.load(os.path.dirname(os.path.dirname(main_path_acc[i]))+'/.hydra/config.yaml')['net']['arch']
        data_exp['scales'] = conf['scales']
        data.append(data_exp)
    np.save('main_data',data)
