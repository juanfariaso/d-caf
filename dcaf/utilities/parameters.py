"""Configuration parameters for the community codes
"""
# TODO: I am not sure how to handle the configuration yet.. lets decide after
# the script gets more complex
from amuse.units import nbody_system, units
from amuse.units import units
from amuse.units.quantities import Quantity

class PetarConfig:
    """Configuration for the PeTar stellar dynamics code.

    Attributes:
        theta:
            Tree opening angle (smaller = more accurate, slower).
        dt_soft:
            Optional soft step, e.g. `0.001 | units.Myr`.
        redirection:
            'none', 'file', or 'stdout'.
        number_of_workers:
            Number of PeTar workers.
        r_bin:
            PeTar binary regularization radius.
        r_out:
            PeTar outer cutoff radius.
    """
    def __init__(self, **kw: object) -> None:
        self.theta: float = 0.5
        #self.dt_soft = 0.125 | units.kyr
        self.dt_soft: Quantity = 2**(-15) | nbody_system.time
        self.redirection: str = "file"
        self.number_of_workers: int = 1
        self.r_bin: Quantity = 100 | units.au #binary regularization limit
        self.r_out: Quantity = 0.03 | units.parsec


class BridgeConfig:
    """Bridge coupling configuration.

    Attributes:
        timestep:
            Coupling timestep. `None` uses the effective PeTar soft timestep.
        use_threading:
            Whether Bridge uses threading.
        verbose:
            Whether Bridge writes verbose output.
    """
    def __init__(self, **kw: object) -> None:
        self.timestep: Quantity = None#, 0.001 | units.Myr #interaction timestep
        self.use_threading: bool = False
        self.verbose: bool = True

class GasConfig:
    """Placeholder configuration for background-gas models."""
    def __init__(self, **kw: object) -> None:
        pass


def get_default_configuration() -> dict:
    """Returns the default configuration for petar bridge and a placeholder for
    background gas.
    Users can start from here and modify accordingly.

    Returns:
        (dict): Dictionary containing default `petar`, `bridge`,
            and `gas` configuration objects.
    """
    return dict( petar = PetarConfig() , bridge = BridgeConfig(), 
                gas = GasConfig()  )
