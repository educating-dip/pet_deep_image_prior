import numpy as np

def normalize(x, inplace=False):
    # Exploding pixel at edge of FOV we need to ignore...
    #mask = np.zeros_like(x)
    #mask[:, 50:201, 50:201] = 1
    #x = mask*x
    if inplace:
        x -= x.min()
        x /= x.max()
    else:
        x = x - x.min()
        x = x / x.max()
    return x

class ComputeImageMetrics:

    def __init__(self, emissions, ROIs_a, ROIs_b):

        self.emissions = emissions
        self.ROIs_a = ROIs_a
        self.ROIs_b = ROIs_b
    
    def _compute_std(self, x):
            # STANDARD DEVIATION
            # abar = ROI average uptake
            # Ka = number of ROIs
            # bbar = background average uptake
            # Kb = number of background ROIs
            # CRC = 1/R \sum_{r=1}^{R} (abar/bbar - 1)/(atrue/btrue - 1)
            return np.std(x[np.nonzero(
                self.ROIs_b
                )]
            )

    def _compute_crc(self, x):
            # CONTRAST RECOVERY COEFFICIENT
            # abar = ROI average uptake
            # Ka = number of ROIs
            # bbar = background average uptake
            # Kb = number of background ROIs
            # CRC = 1/R \sum_{r=1}^{R} (abar/bbar - 1)/(atrue/btrue - 1)
            CRCval = 0
            btrue = self.emissions[-1]
            for i in range(len(self.ROIs_a)):
                abar = np.mean(
                    x[np.nonzero(self.ROIs_a[i]
                    )]
                )
                bbar = np.mean(x[np.nonzero(
                    self.ROIs_b
                    )]
                )
                atrue = self.emissions[i]
                CRCval += (abar / bbar - 1) / (atrue / btrue - 1)
            return CRCval / len(self.ROIs_a)

    def get_all_metrics(self, x):

        return self._compute_crc(x), self._compute_std(x)
