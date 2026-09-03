"""
Star formation decision making. 

This module defines the base class `StarFormationFramework` which handles the
star formation decision making such as:
- How fast stars form (star formation rate). 
- The initial phase-space coordinates of the new stars stars.
- The new stars phase-space coordinates 
- How the background gas potential changes as new stars form (e.g. via star formation efficiency)

Along with any other more complex formation mechanism the user may want to include.

The typical workflow will be by the User creating an inherited class using
`StarFormationFramework` as parent class and overwriting the appropriate
functions.

"""
from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING, Literal
import numpy as np
from amuse.units import units
from amuse.units.quantities import Quantity
from amuse.datamodel.particles import Particles
#pop would do, but this is more efficient
from collections import deque

if TYPE_CHECKING:
    from dcaf.backgroundgas.base import BackgroundPotential

class StarFormationFramework:
    """Base star-formation framework class.

    The user should rewrite some functions of this class. The default behaviour
    is as follows:

    By default, `star_formation_rate="infty"`, so all target stars are
    scheduled as one formation event at time zero. The default `form_stars()`
    implementation returns the scheduled particles unchanged: their masses,
    positions, and velocities are taken directly from `target_stars`.

    For a finite star-formation rate, the framework schedules systems in the
    order they appear in `target_stars`. Stars with the same positive
    `system_id` are kept together; stars without a positive `system_id` are
    treated as single objects. The first event contains at least `nstart`
    stars, without splitting a system. Later systems are scheduled from their
    cumulative stellar mass divided by the supplied star-formation rate, and
    events closer than `dt_tolerance` are merged.

    Args:
        target_stars: The list of stars that will be added to the simulation.
            The order of appearance will be the same as this set.
        star_formation_rate: The star-formation rate at which new stars appear.
            If it is an AMUSE quantity, for example in `MSun / Myr`, the rate
            is constant through the simulation. If it is `"infty"`, all stars
            are added at the beginning of the simulation.
        nstart: The number of stars to form initially. The code advances the
            necessary time, derived from the star-formation rate, to reach the
            required mass for these initial stars.
        background_gas: A `dcaf.backgroundgas.BackgroundPotential` subclass.
            This class controls how the background gas evolves and reacts to
            newly formed stars, and provides the additional acceleration felt
            by stellar particles during the simulation.
        dt_tolerance: Formation events separated by less than this time are
            merged into one event.

    Notes:
        A typical user extension overrides `form_stars(active_stars)`. This
        method provides the new AMUSE particles to add at each scheduled
        formation event. An external star generator can be used here to tune
        coordinates based on the existing stars.

        The `background-gas` model is passed to Bridge to handle gas-star
        interactions. It must provide the appropriate `get_gravity_at_point()`
        method. Custom formation models can also use
        `get_1d_velocity_dispersion_at_point()` or `get_mass_inside_radius()`
        from the background potential to make decisions.

        Tabulated star-formation histories are not implemented yet.

    Attributes:

    """
    # TODO: implement tabulated star formation rates in StarFormationFramework
    def __init__(
        self,
        target_stars: Particles,
        star_formation_rate: Quantity | str  = "infty",
        nstart: int = 2,
        background_gas: BackgroundPotential | None = None,
        dt_tolerance: Quantity = 1e-3 | units.Myr,
    ):
        self.target_stars = target_stars
        self.star_formation_rate = star_formation_rate
        self.nstart = nstart
        self.background_gas = background_gas
        self.dt_tolerance = dt_tolerance

        self.schedule_formation()

    def get_next_formation_time(self) -> Quantity | None:
        """Return the time of the next scheduled formation event.

        Returns:
            (Quantity): The next formation time as an AMUSE quantity
            (None): no formation events remain.
        """
        return self.__next_formation_time

    def get_last_formation_time(self) -> Quantity | None:
        """Return the final time in the current formation schedule.
        
        Notes:
            This function is used for restarting purposes. Restart is only
            implemented if all `target_stars` were already formed.

        Returns:
            (Quantity): The final scheduled formation time.
            (None): No formation events remain.
        """
        next_t = self.get_next_formation_time()

        if next_t is None:
            return None

        if len(self.formation_times) == 0:
            return next_t

        return self.formation_times[-1]


    def extract_next_event(self) -> Particles:
        """Retrieve the next scheduled formation event and advance the schedule.

        This method updates the framework clock and prepares the following
        formation event. Custom `form_stars()` implementations should call it
        exactly once for each event they handle.

        Returns:
            (Particles): The particle set scheduled for the current formation
                event.
        """
        new_stars = self.__next_stars
        self.model_time = self.__next_formation_time
        self.__setup_next_event()

        return new_stars

    def schedule_formation(self, t0: Quantity = 0 | units.Myr) -> None:
        """Build the internal star-formation schedule.

        Instantaneous formation creates one event containing all target stars
        at `t0`. For a finite rate, the formation time of each system is its
        cumulative stellar mass divided by the rate, offset by `t0`.

        Args:
            t0: Time assigned to the first instantaneous event, or the time
                offset applied to a finite-rate formation schedule.

        Raises:
            ValueError: The star-formation rate is not a positive, finite AMUSE
                quantity in mass per time.
            Exception: The target catalogue contains fewer stars than `nstart`
                for a finite-rate schedule.

        Notes:
            Stars sharing a positive `system_id` are scheduled together.
            Missing or non-positive `system_id` values identify single stars.
            The first event reaches or exceeds `nstart` without splitting a
            system. Events separated by less than `dt_tolerance` are merged.
        """
        sfr = self.star_formation_rate
        stars = self.target_stars

        if len(stars) == 0:
            self.formation_sequence, self.formation_times = deque([]), deque([])
            return

        # Instantaneous modes
        if (sfr is None or
            (isinstance(sfr, str) and sfr.lower() in {"infty", "inf"})):
            self.formation_sequence = deque([stars.copy()])
            self.formation_times = deque([t0])
            return

        if not hasattr(sfr, "unit"):
            raise ValueError(
                "SFR must be an AMUSE quantity with units of MSun/Myr"
            )

        sfr_val = sfr.value_in(units.MSun / units.Myr)
        if not isfinite(sfr_val) or sfr_val <= 0.0:
            raise ValueError(
                "Invalid SFR [%S], must be positive and finite" % sfr_val
            )

        if len(stars) < self.nstart:
            raise Exception('Framework must contain at least enough stars for \
            the first batch of nstart = [%i] stars' % self.nstart)

        # Build system groups in order of first appearance.
        raw_ids = stars.system_id if hasattr(stars, "system_id") else None

        groups = {}
        for i in range(len(stars)):
            sid = raw_ids[i] if raw_ids is not None else -1
            key = sid if sid > 0 else -(i + 1)   # singleton fallback
            groups.setdefault(key, []).append(i)

        ordered_keys = list(groups)

        system_batches = [stars[groups[key]].copy() for key in ordered_keys]
        system_sizes = [len(batch) for batch in system_batches]
        system_masses = [batch.mass.sum() for batch in system_batches]

        # One formation time per system.
        cum_mass = []
        mtot = 0 | units.MSun
        for m in system_masses:
            mtot += m
            cum_mass.append(mtot)

        per_system_times = [t0 + m / sfr for m in cum_mass]

        # First batch: keep adding whole systems until we reach/exceed nstart stars.
        first_batch = Particles()
        nfirst = 0
        ifirst = 0

        for i, batch in enumerate(system_batches):
            first_batch.add_particles(batch)
            nfirst += system_sizes[i]
            ifirst = i
            if nfirst >= self.nstart:
                break

        first_time = per_system_times[ifirst]

        formation_sequence = [first_batch]
        formation_times = [first_time]

        # Remaining systems: merge only if times are within dt_tolerance.
        for i in range(ifirst + 1, len(system_batches)):
            this_batch = system_batches[i]
            this_time = per_system_times[i]

            if (formation_times
                    and abs(this_time - formation_times[-1]) < self.dt_tolerance):
                formation_sequence[-1].add_particles(this_batch)
            else:
                formation_sequence.append(this_batch.copy())
                formation_times.append(this_time)

        self.formation_sequence = deque(formation_sequence)
        self.formation_times = deque(formation_times)

        self.__setup_next_event()

    def __setup_next_event(self) -> None:
        """Promote the next queued formation event to the active event."""
        if len(self.formation_sequence) > 0:
            newstars = self.formation_sequence.popleft()
            formation_time = self.formation_times.popleft()
            self.__next_formation_time = formation_time
            self.__next_stars = newstars
        else:
            self.__next_formation_time = None
            self.__next_stars = None

    def get_velocity_dispersion_at_point(
        self,
        x: Quantity,
        y: Quantity,
        z: Quantity,
    ) -> Quantity:
        """Return the one-dimensional velocity dispersion at a position.

        This is for phase-space decisions. 
        If the `background_gas` object is present it returns this same function
        but called on `backgroundgas.get_1d_velocity_dispersion_at_point`.

        If its not present it returns the 1d velocity dispersion of the stars.

        Args:
            x: Position along the x-axis.
            y: Position along the y-axis.
            z: Position along the z-axis.

        Returns:
            (Quantity): 1d velocity dispersion of the environment at `x`,`y`,`z`
        """
        if self.background_gas:
            return self.background_gas.get_1d_velocity_dispersion_at_point(x,y,z)
        else:
            # fallback to current stars velocity dispersion
            vx = self.target_stars.vx.std()
            vy = self.target_stars.vy.std()
            vz = self.target_stars.vz.std()
            return np.mean([vx,vy,vz])

    def form_stars(self, active_stars: Particles = Particles()) -> Particles:
        """Return the stars for the next formation event.

        The base implementation returns the scheduled particle set unchanged.
        It is intended for target lists that already contain the desired
        masses, positions, and velocities.


        Args:
            active_stars: Stars currently active in the N-body simulation. The
                base implementation does not use this value.

        Returns:
            (Particles): The particle set to add for the next formation event.

        Notes:
            Subclasses can override this method to generate positions or
            velocities from `active_stars`. 

            An override **must** call `extract_next_event()` to advance the
            formation schedule.
        """

        new_stars = self.extract_next_event()

        return new_stars
