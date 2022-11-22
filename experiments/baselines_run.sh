python baselines.py -m prior=None
python baselines.py -m prior=QP prior.penalty_factor=0.002,0.001,0.00075,0.0005,0.00025,0.0001
python baselines.py -m prior=QP prior.penalty_factor=0.002,0.001,0.00075,0.0005,0.00025,0.0001, prior.kappa=True prior.initial=True
python baselines.py -m prior=RDP prior.penalty_factor=0.2,0.1,0.075,0.05,0.025,0.01
python baselines.py -m prior=RDP prior.penalty_factor=0.2,0.1,0.075,0.05,0.025,0.01 prior.kappa=True prior.initial=True
python baselines.py -m prior=RDP prior.penalty_factor=2,1.5,1,0.5,0.25,0.1 prior.kappa=True prior.initial=True

python baselines.py -m dataset=3D_high prior=None
python baselines.py -m dataset=3D_high prior=QP prior.penalty_factor=0.002,0.001,0.00075,0.0005,0.00025,0.0001
python baselines.py -m dataset=3D_high prior=QP prior.penalty_factor=0.002,0.001,0.00075,0.0005,0.00025,0.0001, prior.kappa=True prior.initial=True
python baselines.py -m dataset=3D_high prior=RDP prior.penalty_factor=0.2,0.1,0.075,0.05,0.025,0.01
python baselines.py -m dataset=3D_high prior=RDP prior.penalty_factor=0.2,0.1,0.075,0.05,0.025,0.01 prior.kappa=True prior.initial=True
python baselines.py -m dataset=3D_high prior=RDP prior.penalty_factor=2,1.5,1,0.5,0.25,0.1 prior.kappa=True prior.initial=True