import numpy as np
import torch

roi_path = "/home/user/sirf/D690XCATnonTOF/ROI_volumes/3D"
roi_names = ["AbdominalWall","Liver","Lung","Spine"]

AbdominalWallLesionROI = np.load(roi_path + f"/ROI_AbdominalWallTumour.npy")
AbdominalWallLesionTrue = 1682616.375
AbdominalWallBackgroundROI = np.load(roi_path + f"/ROI_AbdominalWallBackground.npy")
AbdominalWallBackgroundTrue = 1053135.375


LiverLesionROI = np.load(roi_path + f"/ROI_LiverTumour.npy")
LiverLesionTrue = 1758343.625
LiverBackgroundROI = np.load(roi_path + f"/ROI_LiverBackground.npy")
LiverBackgroundTrue = 1351406.875

LungLesionROI = np.load(roi_path + f"/ROI_LungTumour.npy")
LungLesionTrue = 911971.4375
LungBackgroundROI = np.load(roi_path + f"/ROI_LungBackground.npy")
LungBackgroundTrue = 446086.0

SpineLesionROI = np.load(roi_path + f"/ROI_SpineTumour.npy")
SpineLesionTrue = 1226146.875
SpineBackgroundROI = np.load(roi_path + f"/ROI_SpineBackground_2.npy")
SpineBackgroundTrue = 788572.1875#762471.4375#


#a_true = [1682616.375,1742961.0,1758343.625,911971.4375,1226146.875]
#b_true = [1053135.375, 1030290.5625, 1351406.875, 446086.0, 762471.4375]
import sirf.STIR as pet
from glob import glob
import os

def get_betas_from_dirs(dirs):
    betas = []
    for i in range(len(dirs)):
        betas.append(float(dirs[i].rsplit("=")[-1][:-1]))
    return betas

def get_epochs_from_files(files):
    epochs = []
    for i in range(len(files)):
        epochs.append(float(files[i].rsplit("_")[-1][:-5]))
    return epochs

def calculate_CRC(vol, a_roi, b_roi, a_true, b_true):
    abar_bbar = np.mean(vol[np.nonzero(a_roi)])/np.mean(vol[np.nonzero(b_roi)])
    return (abar_bbar - 1)/(a_true/b_true - 1)

def calculate_beta_sweep_CRC_STDDEV(path):
    test = glob(path+"/*/", recursive = True)
    dirs = []
    for i in range(len(test)):
        if os.path.isfile(test[i]+"/Final_Volume.hv"):
            dirs.append(test[i])
    betas = get_betas_from_dirs(dirs)
    dirs = [dir for _, dir in sorted(zip(betas, dirs))]
    betas = sorted(betas)
    AbdominalWallCRC = []
    LiverCRC = []
    LungCRC = []
    SpineCRC = []
    AbdominalWallSTDDEV = []
    LiverSTDDEV = []
    LungSTDDEV = []
    SpineSTDDEV = []
    for i in range(len(dirs)):
        x =  pet.ImageData(dirs[i]+"/Final_Volume.hv").as_array()
        AbdominalWallCRC.append(calculate_CRC(x, AbdominalWallLesionROI,AbdominalWallBackgroundROI,AbdominalWallLesionTrue,AbdominalWallBackgroundTrue))
        LiverCRC.append(calculate_CRC(x,LiverLesionROI,LiverBackgroundROI,LiverLesionTrue,LiverBackgroundTrue))
        LungCRC.append(calculate_CRC(x,LungLesionROI,LungBackgroundROI,LungLesionTrue,LungBackgroundTrue))
        SpineCRC.append(calculate_CRC(x,SpineLesionROI,SpineBackgroundROI,SpineLesionTrue,SpineBackgroundTrue))
        """ AbdominalWallCRC.append(calculate_CRC(x, AbdominalWallLesionROI,LiverBackgroundROI,AbdominalWallLesionTrue,LiverBackgroundTrue))
        LiverCRC.append(calculate_CRC(x,LiverLesionROI,LiverBackgroundROI,LiverLesionTrue,LiverBackgroundTrue))
        LungCRC.append(calculate_CRC(x,LungLesionROI,LiverBackgroundROI,LungLesionTrue,LiverBackgroundTrue))
        SpineCRC.append(calculate_CRC(x,SpineLesionROI,LiverBackgroundROI,SpineLesionTrue,LiverBackgroundTrue)) """
        AbdominalWallSTDDEV.append(np.std(x[np.nonzero(AbdominalWallBackgroundROI)]))
        LiverSTDDEV.append(np.std(x[np.nonzero(LiverBackgroundROI)]))
        LungSTDDEV.append(np.std(x[np.nonzero(LungBackgroundROI)]))
        SpineSTDDEV.append(np.std(x[np.nonzero(SpineBackgroundROI)]))
    CRCs = [AbdominalWallCRC, LiverCRC, LungCRC, SpineCRC]
    STDDEVs = [AbdominalWallSTDDEV, LiverSTDDEV, LungSTDDEV, SpineSTDDEV]
    #STDDEVs = [LiverSTDDEV,LiverSTDDEV,LiverSTDDEV,LiverSTDDEV,LiverSTDDEV]
    return CRCs, STDDEVs, betas


def calculate_dip_CRC_STDDEV(x):
    AbdominalWallCRC = []
    LiverCRC = []
    LungCRC = []
    SpineCRC = []
    AbdominalWallSTDDEV = []
    LiverSTDDEV = []
    LungSTDDEV = []
    SpineSTDDEV = []
    AbdominalWallCRC.append(calculate_CRC(x, AbdominalWallLesionROI,AbdominalWallBackgroundROI,AbdominalWallLesionTrue,AbdominalWallBackgroundTrue))
    LiverCRC.append(calculate_CRC(x,LiverLesionROI,LiverBackgroundROI,LiverLesionTrue,LiverBackgroundTrue))
    LungCRC.append(calculate_CRC(x,LungLesionROI,LungBackgroundROI,LungLesionTrue,LungBackgroundTrue))
    SpineCRC.append(calculate_CRC(x,SpineLesionROI,SpineBackgroundROI,SpineLesionTrue,SpineBackgroundTrue))
    AbdominalWallSTDDEV.append(np.std(x[np.nonzero(AbdominalWallBackgroundROI)]))
    LiverSTDDEV.append(np.std(x[np.nonzero(LiverBackgroundROI)]))
    LungSTDDEV.append(np.std(x[np.nonzero(LungBackgroundROI)]))
    SpineSTDDEV.append(np.std(x[np.nonzero(SpineBackgroundROI)]))
    CRCs = [AbdominalWallCRC, LiverCRC, LungCRC, SpineCRC]
    STDDEVs = [AbdominalWallSTDDEV, LiverSTDDEV, LungSTDDEV, SpineSTDDEV]
    return [CRCs, STDDEVs]

def calculate_dip_beta_sweep_CRC_STDDEV(path):
    test = glob(path+"/*/", recursive = True)
    dirs = []
    for i in range(len(test)):
        if os.path.isfile(test[i]+"/Final_best_volume.torch"):
            dirs.append(test[i])
    betas = get_betas_from_dirs(dirs)
    dirs = [dir for _, dir in sorted(zip(betas, dirs))]
    betas = sorted(betas)
    AbdominalWallCRC = []
    LiverCRC = []
    LungCRC = []
    SpineCRC = []
    AbdominalWallSTDDEV = []
    LiverSTDDEV = []
    LungSTDDEV = []
    SpineSTDDEV = []
    for i in range(len(dirs)):
        x =  torch.load(dirs[i]+"/Final_best_volume.torch",torch.device('cpu')).numpy()
        AbdominalWallCRC.append(calculate_CRC(x, AbdominalWallLesionROI,AbdominalWallBackgroundROI,AbdominalWallLesionTrue,AbdominalWallBackgroundTrue))
        LiverCRC.append(calculate_CRC(x,LiverLesionROI,LiverBackgroundROI,LiverLesionTrue,LiverBackgroundTrue))
        LungCRC.append(calculate_CRC(x,LungLesionROI,LungBackgroundROI,LungLesionTrue,LungBackgroundTrue))
        SpineCRC.append(calculate_CRC(x,SpineLesionROI,SpineBackgroundROI,SpineLesionTrue,SpineBackgroundTrue))
        AbdominalWallSTDDEV.append(np.std(x[np.nonzero(AbdominalWallBackgroundROI)]))
        LiverSTDDEV.append(np.std(x[np.nonzero(LiverBackgroundROI)]))
        LungSTDDEV.append(np.std(x[np.nonzero(LungBackgroundROI)]))
        SpineSTDDEV.append(np.std(x[np.nonzero(SpineBackgroundROI)]))
    CRCs = [AbdominalWallCRC, LiverCRC, LungCRC, SpineCRC]
    STDDEVs = [AbdominalWallSTDDEV, LiverSTDDEV, LungSTDDEV, SpineSTDDEV]
    return CRCs, STDDEVs, betas
    
def calculate_dip_run_CRC_STDDEV(path):
    files = glob(path+"/Best_volume_epoch_*.torch", recursive = True)
    epochs = get_epochs_from_files(files)
    files = [file for _, file in sorted(zip(epochs, files))]
    epochs = sorted(epochs)
    epochs = epochs
    AbdominalWallCRC = []
    LiverCRC = []
    LungCRC = []
    SpineCRC = []
    AbdominalWallSTDDEV = []
    LiverSTDDEV = []
    LungSTDDEV = []
    SpineSTDDEV = []
    for i in range(len(files)):
        x =  torch.load(files[i],torch.device('cpu')).numpy()
        AbdominalWallCRC.append(calculate_CRC(x, AbdominalWallLesionROI,AbdominalWallBackgroundROI,AbdominalWallLesionTrue,AbdominalWallBackgroundTrue))
        LiverCRC.append(calculate_CRC(x,LiverLesionROI,LiverBackgroundROI,LiverLesionTrue,LiverBackgroundTrue))
        LungCRC.append(calculate_CRC(x,LungLesionROI,LungBackgroundROI,LungLesionTrue,LungBackgroundTrue))
        SpineCRC.append(calculate_CRC(x,SpineLesionROI,SpineBackgroundROI,SpineLesionTrue,SpineBackgroundTrue))
        AbdominalWallSTDDEV.append(np.std(x[np.nonzero(AbdominalWallBackgroundROI)]))
        LiverSTDDEV.append(np.std(x[np.nonzero(LiverBackgroundROI)]))
        LungSTDDEV.append(np.std(x[np.nonzero(LungBackgroundROI)]))
        SpineSTDDEV.append(np.std(x[np.nonzero(SpineBackgroundROI)]))
    CRCs = [AbdominalWallCRC, LiverCRC, LungCRC, SpineCRC]
    STDDEVs = [AbdominalWallSTDDEV, LiverSTDDEV, LungSTDDEV, SpineSTDDEV] 
    #STDDEVs = [LiverSTDDEV,LiverSTDDEV,LiverSTDDEV,LiverSTDDEV,LiverSTDDEV]
    return CRCs, STDDEVs, epochs
