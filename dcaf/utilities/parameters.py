# TODO: I am not sure how to handle the configuration yet.. lets decide after
# the script gets more complex
from amuse.units import nbody_system, units
from amuse.units import units

class PetarConfig:
    """Configuration for the PeTar stellar dynamics code.

    Attributes
    ----------
    theta : float
        Tree opening angle (smaller = more accurate, slower).
    dt_soft : object | None
        Optional soft step (AMUSE units), e.g. `0.001 | units.Myr`.
    redirection : str
        'none', 'file', or 'stdout'.
    extra_options : dict
        Additional parameters forwarded to `petar.parameters` if they exist.
    """
    def __init__(self,**kw):
        self.theta = 0.5
        #self.dt_soft = 0.125 | units.kyr
        self.dt_soft = 2**(-15) | nbody_system.time
        self.redirection = "file"
        self.number_of_workers = 1
        self.r_bin = 100 | units.au #binary regularization limit
        self.r_out = 0.03 | units.parsec


class BridgeConfig:
    """Bridge coupling configuration."""
    def __init__(self,**kw):
        self.timestep = None#, 0.001 | units.Myr #interaction timestep
        self.use_threading = False
        self.verbose = True

class GasConfig:
    """Bridge coupling configuration."""
    def __init__(self,**kw):
        pass

class StellarEvolutionConfig:
    """Stellar evolution configuration."""
    def __init__(self, **kw):
        self.metallicity = 0.02


def get_default_configuration():
    return dict( petar = PetarConfig() , 
                bridge = BridgeConfig(), 
                gas = GasConfig(),
                stellar_evolution = StellarEvolutionConfig() 
                )
