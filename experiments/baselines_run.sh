python baselines.py -m prior=OSEM

python baselines.py -m prior=MAPEM_QP prior.penalty_factor=0.005,0.001,0.0075,0.0005,0.00025,0.0001,0.00001
python baselines.py -m prior=MAPEM_QP prior.penalty_factor=0.005,0.001,0.0075,0.0005,0.00025,0.0001,0.00001 prior.kappa=True prior.initial=True
python baselines.py -m prior=MAPEM_RDP prior.penalty_factor=0.5,0.1,0.075,0.05,0.025,0.01,0.001
python baselines.py -m prior=MAPEM_RDP prior.penalty_factor=0.5,0.1,0.075,0.05,0.025,0.01,0.001 prior.kappa=True prior.initial=True