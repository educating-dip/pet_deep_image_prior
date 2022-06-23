import sirf.STIR as pet

class PriorClass(object):
    def __init__(self,cfg,dataset):
        if cfg.prior.name == 'None':
            NoPrior(dataset.initial,
                dataset.objective_function
                )
        elif cfg.prior.name == 'QP':
            QuadraticPrior(dataset.initial,
                dataset.objective_function,
                cfg.prior.kappa,
                cfg.dataset.kappa,
                cfg.prior.penalty_factor
                )
        elif cfg.prior.name == 'RDP':
            RelativeDifferencePrior(dataset.initial,
                dataset.objective_function,
                cfg.prior.kappa,
                cfg.dataset.kappa,
                cfg.prior.penalty_factor,
                cfg.prior.gamma
                )
        else:
            raise NotImplementedError


def NoPrior(initial,
        objective_function
        ):
        print("Just plain old OSEM")
        objective_function.set_up(initial)

def QuadraticPrior(initial,
                objective_function,
                kappa_use,
                kappa_path,
                penalty_factor
                ):
        prior = pet.QuadraticPrior()
        print('using Quadratic prior...')
        prior.set_penalisation_factor(penalty_factor)
        if kappa_use:
            kappa = pet.ImageData(kappa_path)
            prior.set_kappa(initial.clone().fill(kappa))
        prior.set_up(initial)
        objective_function.set_prior(prior)
        objective_function.set_up(initial)

def RelativeDifferencePrior(
                objective_function,
                initial_use,
                initial_path,
                kappa_use,
                kappa_path,
                penalty_factor,
                gamma
                ):
        prior = pet.RelativeDifferencePrior()
        print('using Relative Difference prior...')
        prior.set_penalisation_factor(penalty_factor)
        prior.set_gamma(gamma)
        if initial_use:
            initial = pet.ImageData(initial_path)
        if kappa_use:
            kappa = pet.ImageData(kappa_path)
            prior.set_kappa(initial.clone().fill(kappa))
        prior.set_up(initial)
        objective_function.set_prior(prior)
        objective_function.set_up(initial)