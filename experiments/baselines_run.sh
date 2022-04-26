python baselines.py -m dataset=2D_low,2D_medium,2D_high prior=OSEM

python baselines.py -m dataset=2D_high prior=MAPEM_QP prior.penalty_factor=1e-7,1e-6,5e-6,1e-5 prior.kappa=False prior.initial=False
python baselines.py -m dataset=2D_high prior=MAPEM_QP prior.penalty_factor=1e-4,1e-3,5e-3,1e-2 prior.kappa=True
python baselines.py -m dataset=2D_high prior=MAPEM_RDP prior.penalty_factor=1e-3,1e-2,5e-2,1e-1 prior.kappa=False prior.initial=False
python baselines.py -m dataset=2D_high prior=MAPEM_RDP prior.penalty_factor=1e-2,1e-1,5e-1,1e0 prior.kappa=True

python baselines.py -m dataset=2D_medium prior=MAPEM_QP prior.penalty_factor=1e-5,1e-4,5e-4,1e-3 prior.kappa=False prior.initial=False 
python baselines.py -m dataset=2D_medium prior=MAPEM_QP prior.penalty_factor=1e-4,1e-3,5e-3,1e-2 prior.kappa=True
python baselines.py -m dataset=2D_medium prior=MAPEM_RDP prior.penalty_factor=1e-3,1e-2,5e-2,1e-1 prior.kappa=False prior.initial=False
python baselines.py -m dataset=2D_medium prior=MAPEM_RDP prior.penalty_factor=1e-2,1e-1,5e-1,1e0 prior.kappa=True

python baselines.py -m dataset=2D_low prior=MAPEM_QP prior.penalty_factor=1e-5,1e-4,5e-4,1e-3 prior.kappa=False prior.initial=False 
python baselines.py -m dataset=2D_low prior=MAPEM_QP prior.penalty_factor=1e-4,1e-3,5e-3,1e-2 prior.kappa=True
python baselines.py -m dataset=2D_low prior=MAPEM_RDP prior.penalty_factor=1e-3,1e-2,5e-2,1e-1 prior.kappa=False prior.initial=False
python baselines.py -m dataset=2D_low prior=MAPEM_RDP prior.penalty_factor=1e-2,1e-1,5e-1,1e0 prior.kappa=True