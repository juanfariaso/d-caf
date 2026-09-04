"""Base classes helpers to define background gas models."""
from amuse.units import units
from amuse.datamodel import Particles
from amuse.units.quantities import Quantity, zero
import numpy as np
import os

class BackgroundPotential:
    """
    Base class for Background Gas potentials used in D-CAF

    Args:
        mtot: Total mass of the potential (interpretation depends on subclass).
        rscale: Scale radius (interpretation depends on subclass).

    Attributes:
        mtot (Quantity): Current total mass of the gas potential.
        rscale (Quantity): Current scale radius of the gas potential.
        logfilename (str): File written by
            [write_output()][dcaf.backgroundgas.base.BackgroundPotential.write_output].
        model_time (Quantity): Current gas-model time.

    """

    def __init__(self, mtot: Quantity, rscale: Quantity) -> None:
        self.mtot = mtot
        self.rscale = rscale

        # What attributes should be saved into a file?
        self.logfile = None
        self.logfilename = 'background_gas.dat'
        self.output_attributes = ['mtot','rscale' ] 
        self.output_units = [ units.MSun, units.parsec  ]
        self.__first_output = True

    # --- Bridge-facing API ---
    def get_potential_at_point(
        self,
        eps: Quantity,
        x: Quantity,
        y: Quantity,
        z: Quantity,
    ) -> Quantity:
        """
        Return the gravitational potential at (x, y, z).

        Args:
            eps: Gravitational softening length.
            x: X coordinates at which to evaluate the potential.
            y: Y coordinates at which to evaluate the potential.
            z: Z coordinates at which to evaluate the potential.

        Returns:
            (Quantity): Gravitational potential at each point.

        Raises:
            NotImplementedError: Always, unless implemented by a subclass.

        """
        raise NotImplementedError

    def get_gravity_at_point(
        self,
        eps: Quantity,
        x: Quantity,
        y: Quantity,
        z: Quantity,
    ) -> tuple[Quantity, Quantity, Quantity]:
        """
        Return the gravitational acceleration vector at (x, y, z).
        Should return (ax, ay, az).

        Args:
            eps: Gravitational softening length.
            x: X coordinates at which to evaluate acceleration.
            y: Y coordinates at which to evaluate acceleration.
            z: Z coordinates at which to evaluate acceleration.

        Returns:
            (tuple[Quantity, Quantity, Quantity]):  X, Y, and Z acceleration
            components.

        Raises:
            NotImplementedError: Always, unless implemented by a subclass.
        """
        raise NotImplementedError

    def get_potential_derivative_at_point(
        self,
        x: Quantity,
        y: Quantity,
        z: Quantity,
    ) -> Quantity:
        """
        Return dPhi/dt at (x, y, z). Optional, used mainly for energy checks.

        Args:
            x: X coordinates at which to evaluate the derivative.
            y: Y coordinates at which to evaluate the derivative.
            z: Z coordinates at which to evaluate the derivative.

        Returns:
            (Quantity): Time derivative of gravitational potential at each point.

        Raises:
            NotImplementedError: Always, unless implemented by a subclass.

        Note:
            If not defined, then [DcafSystem][dcaf.dcaf.DcafSystem] should have
            `track_background_gas_energy = False` to skip this function.
        """
        raise NotImplementedError

    def get_1d_velocity_dispersion_at_point(
        self,
        x: Quantity,
        y: Quantity,
        z: Quantity,
    ) -> Quantity:
        """Return the local one-dimensional velocity dispersion.

        Args:
            x: X coordinates at which to evaluate the dispersion.
            y: Y coordinates at which to evaluate the dispersion.
            z: Z coordinates at which to evaluate the dispersion.

        Returns:
            (Quantity): One-dimensional velocity dispersion at each point.

        Raises:
            NotImplementedError: Always, unless implemented by a subclass.

        Note:
            This method is required only when the gas model is used with
            [StarFormationFramework][dcaf.framework.starformation.StarFormationFramework].
        """
        raise NotImplementedError

    def is_gas_relevant(self, stars: Particles, **kwargs: float) -> bool:
        """Return whether the gas remains dynamically relevant.

        Args:
            stars: [DOCS-REVIEW] Current stellar particles used by the
                model-specific relevance criterion.
            **kwargs: [DOCS-REVIEW] Model-specific relevance-criterion options.

        Returns:
            (bool): [DOCS-REVIEW] Whether the gas should remain active in the
                simulation.

        Raises:
            NotImplementedError: Always, unless implemented by a subclass.
        """
        raise NotImplementedError

    def _ensure_logfile(self) -> None:
        """Open the output file and initialize its header state."""
        #Open logfile if needed. If the file already exists, assume the header
        #was written in a previous session and keep __first_output = False.
        #This is needed when rebuilding dcaf system, because closing bridge
        #calls the cleanup_code here that we need to be clean but it closes 
        #our logfile
        if self.logfile is not None and not self.logfile.closed:
            return

        file_exists = os.path.exists(self.logfilename)
        mode = "a" if file_exists else "w"
        self.logfile = open(self.logfilename, mode)

        if file_exists:
            self.__first_output = False
        else:
            self.__first_output = True

    def get_mass_inside_radius(self, r: Quantity) -> Quantity:
        """
        Return the mass enclosed within radius `r`.

        Args:
            r: Radius or radii at which to evaluate enclosed mass.

        Returns:
            (Quantity): Enclosed gas mass.

        Raises:
            NotImplementedError: Always, unless implemented by a subclass.
        """
        raise NotImplementedError

    def initialize_code(self) -> None:
        """Initialize the AMUSE-compatible gas model time."""
        self.model_time = zero
        pass

    def commit_parameters(self) -> None:
        """Implement AMUSE's parameter-commit hook."""
        pass

    def commit_particles(self) -> None:
        """Implement AMUSE's particle-commit hook."""
        pass

    def cleanup_code(self) -> None:
        """Close the background-gas output file."""
        if self.logfile is not None and not self.logfile.closed:
            self.logfile.close()
        pass

    def evolve_model(self, tend: Quantity) -> None:
        """Advance the gas model to the requested time.

        Args:
            tend: Target gas-model time.
        """
        self.model_time = tend
        pass

    def write_output(self) -> None:
        """Append the configured gas attributes to the output file."""

        self._ensure_logfile()

        if self.__first_output:
            header = ['# Time [Myr]']
            header += [f'{k} [{u}]' if u is not None else k
                       for k,u in zip(self.output_attributes, self.output_units)]
            self.logfile.write(' '.join(header) + '\n')
            self.__first_output = False

        vals = [self.model_time.value_in(units.Myr)]
        for k,u in zip(self.output_attributes,self.output_units):
            v = getattr(self, k)
            try :
                vals.append(v.value_in(u) if u is not None else float(v))
            except:
                vals.append(np.Nan)
        self.logfile.write(' '.join(f'{x:.6g}' for x in vals) + '\n')
        self.logfile.flush()


    def stop(self) -> None:
        """Stop the gas model and close its output file."""
        self.cleanup_code()
