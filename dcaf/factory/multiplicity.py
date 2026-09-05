"""
This module should help introducing multiplicity into an existent star list.

The default method idea is to keep the distribution fixed but group stars
together around a primary. No change to the IMF and only change the positions
and velocity of the companions.

Then, we proceed as follows:

1. We choose which stars become primordial binaries
2. Pick companions by adjusting to a predefined mass ratio distribution
3. Calculate the orbits of the systems based on binary populations, probably a\
user defined function could also work.
4. Put all stars together at the location of the primary, using its coordinates\
as center of mass.

## Hierarchy Format

In the implementation, we refeer to `hierarchy` to store binaries and triples
and work with them. `hierarchy` is just a list of dictionaries. Each entry
corresponds to the particle at the same index in `unresolved_stars`; `members`
contains the indices of that system's component particles in `resolved_stars`.
The entries in `periods`, `semi_major_axes`, `eccentricities`, and `lhat` at a
common index describe the same orbit. In the standard triple workflow, those
lists are ordered from the inner orbit to the newly added outer orbit.

| Key | Value |
| --- | --- |
| `members` | `list[int]` of indices into `resolved_stars`. |
| `periods` | `list[Quantity]` of orbital periods with time dimensions. |
| `semi_major_axes` | `list[Quantity]` of semimajor axes with length dimensions. |
| `eccentricities` | `list[float]` of dimensionless eccentricities. |
| `lhat` | `list[numpy.ndarray]` of dimensionless, three-component unit angular-momentum vectors. |
| `_resolved_particles` | Internal `Particles` retained for recursive pairing; do not modify it. |

Example:
    A hierarchy containing a triple, a binary, and a single system:

    ```python
    hierarchy = [
        {
            # Triple: inner orbit (0, 1), then outer orbit ((0, 1), 2).
            "members": [0, 1, 2],
            "periods": [10 | units.day, 1e4 | units.day],
            "semi_major_axes": [0.1 | units.AU, 10 | units.AU],
            "eccentricities": [0.1, 0.3],
            "lhat": [
                np.array([0.0, 0.0, 1.0]),
                np.array([0.0, 1.0, 0.0]),
            ],
        },
        {
            # Binary: orbit (3, 4).
            "members": [3, 4],
            "periods": [100 | units.day],
            "semi_major_axes": [1 | units.AU],
            "eccentricities": [0.2],
            "lhat": [np.array([1.0, 0.0, 0.0])],
        },
        {
            # Single: no orbital entries.
            "members": [5],
            "periods": [],
            "semi_major_axes": [],
            "eccentricities": [],
            "lhat": [],
        },
    ]
    ```

"""

import numpy

from amuse.lab import Particles, units
from amuse.units.constants import G
from amuse.units.quantities import Quantity
import numpy as np
from dcaf.factory import field_binary_population as fbp


class BinaryPopulation:
    r"""
    Build binary and hierarchical multiple initial conditions from AMUSE
    `Particles`.

    Args:
        nbinaries:  Number of binaries to create. Provide this or
            `population_fraction`, but not both.
        population_fraction:  Fractions of single, binary, and optionally triple
            systems.
        gamma:  Exponent of the mass-ratio weighting $p(q) \propto q^\gamma$.
        q_min:  Minimum allowed companion-to-primary mass ratio.
        mean_period: Mean period of the log-normal period model.
        sigma_logP: Standard deviation of $\log_{10}(P/\mathrm{day})$.
        eccentricities: Eccentricity model: `circular`, `thermal`, or `flat`.
        max_semi_major_axis: Maximum allowed binary semimajor axis. If omitted,
            the stellar half-mass radius is used.
        higher_order_mode: Placement mode for higher-order companions.
            `hierarchical` uses subsystem centres of mass; `primary_centered` places
            companions relative to the primary star.
        companion_mode: Companion source mode: `pool`, pair binaries form the
            exiting stellar list; `create`, select primaries from the list and
            creates their secondaries discarding stars that will not be used
            based on the original total mass. 

    Attributes:
        nbinaries (int): Configured number of binaries.
        population_fraction (list[float]): Configured system-order fractions.
        gamma (float): Mass-ratio weighting exponent.
        q_min (float): Minimum allowed mass ratio.
        mean_period (Quantity): Mean binary period.
        sigma_logP (float): Log-period dispersion.
        eccentricities (str): Eccentricity population model.
        max_semi_major_axis (Quantity): Maximum allowed binary semimajor axis.
        higher_order_mode (str): Higher-order placement mode.
        companion_mode (str): Companion source mode.
    """

    def __init__(
        self,
        nbinaries: int = None,
        population_fraction: list[float] = None,
        gamma: float = 0.0,
        q_min: float = 0.0,
        mean_period: Quantity = 10.0 ** 4.8 | units.day,
        sigma_logP: float = 2.3,
        eccentricities: str = "circular",
        max_semi_major_axis: Quantity = None,
        higher_order_mode: str = "hierarchical",
        companion_mode: str = "pool",
    ) -> None:
        self.nbinaries = nbinaries
        self.population_fraction = population_fraction
        self.gamma = gamma
        self.q_min = q_min
        self.mean_period = mean_period
        self.sigma_logP = sigma_logP
        self.eccentricities = eccentricities
        self.max_semi_major_axis = max_semi_major_axis
        self.higher_order_mode = higher_order_mode

        self.companion_mode = companion_mode
        if self.companion_mode not in ("pool", "create"):
            raise ValueError("companion_mode must be 'pool' or 'create'.")

        if (self.nbinaries is None) == (self.population_fraction is None):
            raise ValueError("Provide exactly one of `nbinaries` or `population_fraction`.")
        if self.population_fraction is not None:
            if len(self.population_fraction) < 2:
                raise ValueError(
                    "population_fraction must contain at least [single_fraction, binary_fraction]."
                )
            if numpy.any(numpy.array(self.population_fraction) < 0.0):
                raise ValueError("population_fraction entries must be non-negative.")
            if numpy.sum(self.population_fraction) <= 0.0:
                raise ValueError("population_fraction must have a positive sum.")
            if len(self.population_fraction) > 3:
                extra = numpy.array(self.population_fraction[3:])
                if numpy.any(extra > 0.0):
                    raise NotImplementedError(
                        "BinaryPopulation currently supports only singles, binaries, and triples."
                    )
        if self.q_min < 0.0 or self.q_min > 1.0:
            raise ValueError("q_min must be between 0 and 1.")
        if self.higher_order_mode not in ("hierarchical", "primary_centered"):
            raise ValueError(
                "higher_order_mode must be 'hierarchical' or 'primary_centered'."
            )

    def get_number_of_binaries(self, stars: Particles) -> int:
        """
        Determine how many binary systems to construct from the configured
        population parameters.

        Args:
            stars: Input stellar particles.

        Returns:
            (int): Number of first-level binaries to construct.
        """
        if self.nbinaries is not None:
            nbinaries = self.nbinaries
        else:
            fractions = numpy.array(self.population_fraction, dtype=float)
            fractions = fractions / fractions.sum()
            mean_multiplicity = numpy.sum(
                fractions * numpy.arange(1, len(fractions) + 1, dtype=float)
            )
            nsystems = float(len(stars)) / mean_multiplicity
            nbinaries = int(numpy.floor(fractions[1] * nsystems))

        if nbinaries < 0:
            raise ValueError("nbinaries must be non-negative.")
        if nbinaries > len(stars) // 2:
            raise ValueError("nbinaries can not exceed len(stars)//2.")

        return nbinaries

    def preprocess_stars(self, stars: Particles) -> Particles:
        """
        This function is meant to be overwritten by inherited classes.
        This function is called inside apply before performing the pairing.
        It purpose is to give flexibility, in case we need to modify
        stars that will go into the binary population. 


        Example case:
            Currently the code pair stars from the pool of existing stars
            (`companion_mode = pool`). 
            However, another option is to create the companions inplace based on a
            given mass ratio, this is what `companion_mode = create` does. 

            In this case, the preprocess_stars function do:

            1) for each primary, assign a mass ratio and create the companion
                
            2) at each primary, link its companion adding it index as a new property.
                
            3) then the choose_companion function (rewritten by the user) can read
            this index and return it so the rest of the class do the rest.

        Args:
            stars: Input stellar particles.

        Returns:
            (Particles): Prepared stellar particles.

        """

        return stars


    def choose_companion(
        self,
        m1: Quantity,
        mass: Quantity,
        equal_mass: bool = False,
    ) -> Quantity:
        r"""
        Select one companion mass for `m1` from the available pool `mass`  using
        the configured mass-ratio distribution $p(q) \propto q^{\gamma}$, where
        $q = m_2/m_1 <= 1$.

        Args:
            m1: Primary-star mass.
            mass: Available companion masses.
            equal_mass: Whether to sample uniformly rather than apply mass-ratio
                weighting (for testing purposes).

        Returns:
            (Quantity): Selected companion mass.
        """
        if len(mass) == 0:
            raise ValueError("No companion masses available.")

        if equal_mass:
            index = numpy.random.randint(len(mass))
            return mass[index]

        if m1 <= mass.min():
            raise Exception("m1 cannot be the smaller mass in the set")

        q = mass.value_in(m1.unit) / m1.value_in(m1.unit)
        valid = (q <= 1.0) & (q >= self.q_min)

        if not numpy.any(valid):
            raise Exception("No available companion masses satisfy q_min <= q <= 1.")

        valid_index = numpy.where(valid)[0]
        q_valid = q[valid]
        weights = q_valid ** self.gamma

        if not numpy.all(numpy.isfinite(weights)):
            raise ValueError("Non-finite weights encountered in mass-ratio sampling.")
        if weights.sum() <= 0:
            raise ValueError("Mass-ratio weights sum to zero.")

        weights = weights / weights.sum()
        choice = numpy.random.choice(valid_index, p=weights)
        return mass[choice]

    def _resolve_max_semi_major_axis(
        self,
        stars: Particles,
        max_semi_major_axis: Quantity = None,
    ) -> Quantity:
        """Return the configured semimajor-axis cap or the half-mass radius."""
        if max_semi_major_axis is None:
            max_semi_major_axis = self.max_semi_major_axis
        if max_semi_major_axis is None:
            max_semi_major_axis = stars.LagrangianRadii(
                mf=[0.5], cm=stars.center_of_mass()
            )[0][0]
        if max_semi_major_axis <= 0 | max_semi_major_axis.unit:
            raise ValueError("max_semi_major_axis must be positive.")
        return max_semi_major_axis

    def get_periods(
        self,
        stars: Particles,
        primary_index: list[int],
        companion_index: list[int],
        max_semi_major_axis: Quantity = None,
        **kwargs: Quantity,
    ) -> Quantity:
        """
        Sample binary periods for the selected systems.

        It draws from the configured log-normal period distribution and rejects
        samples whose Keplerian semimajor axes exceed the configured cap.

        Args:
            stars: Stellar particles being paired.
            primary_index: Indices of primary stars.
            companion_index: Indices of companion stars.
            max_semi_major_axis: Optional semimajor-axis cap. If omitted, uses
                `self.max_semi_major_axis` or the stellar half-mass radius.

        Returns:
            (Quantity): Sampled orbital periods.

        Raises:
            ValueError: If the cap is non-positive or smaller than the orbit
                allowed by the minimum one-day period.
        """
        nbinaries = len(primary_index)
        max_semi_major_axis = self._resolve_max_semi_major_axis(
            stars, max_semi_major_axis
        )
        binary_mass = stars.mass[primary_index] + stars.mass[companion_index]
        Pmin = 1.0 | units.day
        minimum_semi_major_axes = (
            (Pmin.value_in(units.yr) ** 2) * binary_mass.value_in(units.MSun)
        ) ** (1.0 / 3.0) | units.AU
        if numpy.any(minimum_semi_major_axes > max_semi_major_axis):
            raise ValueError(
                "max_semi_major_axis is smaller than the minimum orbit allowed "
                "by Pmin."
            )

        periods = numpy.zeros(nbinaries) | units.day
        pending = list(range(nbinaries))
        while len(pending) > 0:
            logP = numpy.random.normal(
                loc=numpy.log10(self.mean_period.value_in(units.day)),
                scale=self.sigma_logP,
                size=len(pending),
            )
            periods[pending] = 10.0 ** logP | units.day
            semi_major_axes = (
                (periods[pending].value_in(units.yr) ** 2)
                * binary_mass[pending].value_in(units.MSun)
            ) ** (1.0 / 3.0) | units.AU
            invalid = (periods[pending] < Pmin) | (
                semi_major_axes > max_semi_major_axis
            )
            pending = [pending[i] for i in numpy.where(invalid)[0]]

        return periods

    def get_eccentricities(
        self,
        stars: Particles,
        primary_index: list[int],
        companion_index: list[int],
        **kwargs: Quantity,
    ) -> numpy.ndarray:
        """
        Sample binary eccentricities according to the configured eccentricity
        population model.

        Args:
            stars: Stellar particles being paired.
            primary_index: Indices of primary stars.
            companion_index: Indices of companion stars.

        Returns:
            (numpy.ndarray): One eccentricity per binary.
        """
        nbinaries = len(primary_index)

        if self.eccentricities == "thermal":
            return numpy.sqrt(numpy.random.uniform(size=nbinaries))
        if self.eccentricities == "circular":
            return numpy.zeros(nbinaries)
        if self.eccentricities == "flat":
            return numpy.random.uniform(size=nbinaries)
        raise Exception(
            "Eccentricity population '%s' not implemented" % self.eccentricities
        )

    def get_normalized_angular_momentum(
        self,
        stars: Particles,
        primary_index: list[int],
        companion_index: list[int],
        **kwargs: Quantity,
    ) -> numpy.ndarray | None:
        """
        Return per-binary normalized angular-momentum directions.

        The default implementation returns ``None``, meaning that orbit-plane
        orientations are sampled randomly in ``make_binaries()``.

        When provided, the return value must be a NumPy-like array with shape
        ``(nbinaries, 3)``, one row per binary. Rows with non-finite values or
        zero norm are ignored and fall back to random orientation.

        Note:
            The resulting angular momentum orientations do not need to be
            normalized, the code only used it directions.

        Args:
            stars: Stellar particles being paired.
            primary_index: Indices of primary stars.
            companion_index: Indices of companion stars.

        Returns:
            (numpy.ndarray | None):  Unit angular-momentum
                vectors, or `None` for random orientations.
        """
        return None

    def select_higher_order_companion(
        self,
        primary_index: int,
        stars: Particles,
        hierarchy: list[dict],
        available_indexes: list[int],
        **kwargs: Quantity,
    ) -> int | None:
        """
        Select one higher-order companion for the given primary system.

        The default implementation only supports the minimal hierarchical
        channel `multiple + single -> triple`. The primary system must already
        contain exactly two members, and the chosen companion must be a single
        unresolved object. However, subclasses may want to do more complex
        implementations.

        Args:
            primary_index: Index of the existing multiple system.
            stars: Unresolved stellar or system particles.
            hierarchy: Current <a href="#dcaf.factory.multiplicity--hierarchy-format">hierarchy dictionaries</a>.
            available_indexes: Candidate unresolved-system indices.

        Returns:
            (int | None): Selected companion index, if available.
        """
        if len(hierarchy[primary_index]["members"]) != 2:
            return None

        candidates = [
            i for i in available_indexes
            if i != primary_index and len(hierarchy[i]["members"]) == 1
        ]
        if len(candidates) == 0:
            return None

        companion_mass = self.choose_companion(
            stars.mass[primary_index],
            stars.mass[candidates],
            equal_mass=False,
        )
        for index in candidates:
            if stars.mass[index] == companion_mass:
                return index
        return None

    def get_higher_order_periods(
        self,
        stars: Particles,
        primary_index: list[int],
        companion_index: list[int],
        hierarchy: list[dict],
        min_semimajor_ratio: float = 10.0,
        max_semi_major_axis: Quantity = None,
        **kwargs: Quantity,
    ) -> Quantity:
        """
        Sample outer periods for hierarchical pairings.

        By default this reuses `get_periods(...)` and rejects samples until the
        resulting outer semimajor axis is at least `min_semimajor_ratio` times
        the last stored semimajor axis of the primary hierarchy branch, while
        remaining below `max_semi_major_axis`.

        Args:
            stars: Unresolved stellar or system particles.
            primary_index: Indices of existing multiple systems.
            companion_index: Indices of selected companions.
            hierarchy: Current <a href="#dcaf.factory.multiplicity--hierarchy-format">hierarchy dictionaries</a>.
            min_semimajor_ratio: Required outer-to-inner semimajor-axis ratio.
            max_semi_major_axis: Maximum allowed outer semimajor axis. If
                omitted, uses the configured cap or stellar half-mass radius.

        Returns:
            (Quantity): Sampled outer orbital periods.

        Raises:
            ValueError: If no outer orbit can satisfy the semimajor-axis-ratio
                constraint.
        """
        total_mass = stars.mass[primary_index] + stars.mass[companion_index]
        max_semi_major_axis = self._resolve_max_semi_major_axis(
            stars, max_semi_major_axis
        )
        periods = self.get_periods(
            stars,
            primary_index,
            companion_index,
            max_semi_major_axis=max_semi_major_axis,
            **kwargs,
        ).copy()

        pending = list(range(len(primary_index)))
        while len(pending) > 0:
            semi_major_axes = (
                (periods[pending].value_in(units.yr) ** 2)
                * total_mass[pending].value_in(units.MSun)
            ) ** (1.0 / 3.0) | units.AU

            keep_pending = []
            for local_i, system_i in enumerate(pending):
                primary_hierarchy = hierarchy[primary_index[system_i]]
                if len(primary_hierarchy["semi_major_axes"]) == 0:
                    continue
                inner_a = primary_hierarchy["semi_major_axes"][-1]
                if min_semimajor_ratio * inner_a > max_semi_major_axis:
                    raise ValueError(
                        "No valid higher-order outer orbit fits within the current "
                        "maximum semimajor-axis cap."
                    )
                if semi_major_axes[local_i] < min_semimajor_ratio * inner_a:
                    keep_pending.append(system_i)

            if len(keep_pending) == 0:
                break

            periods[keep_pending] = self.get_periods(
                stars,
                [primary_index[i] for i in keep_pending],
                [companion_index[i] for i in keep_pending],
                max_semi_major_axis=max_semi_major_axis,
                **kwargs,
            )
            pending = keep_pending

        return periods

    def get_higher_order_eccentricities(
        self,
        stars: Particles,
        primary_index: list[int],
        companion_index: list[int],
        hierarchy: list[dict],
        **kwargs: Quantity,
    ) -> numpy.ndarray:
        """
        Sample eccentricities for hierarchical pairings.

        The default implementation reuses `get_eccentricities(...)`.

        Args:
            stars: Unresolved stellar or system particles.
            primary_index: Indices of existing multiple systems.
            companion_index: Indices of selected companions.
            hierarchy: Current <a href="#dcaf.factory.multiplicity--hierarchy-format">hierarchy dictionaries</a>.

        Returns:
            (numpy.ndarray): One outer eccentricity per pairing.
        """
        return self.get_eccentricities(
            stars, primary_index, companion_index, **kwargs
        )

    def get_higher_order_normalized_angular_momentum(
        self,
        stars: Particles,
        primary_index: list[int],
        companion_index: list[int],
        hierarchy: list[dict],
        **kwargs: Quantity,
    ) -> numpy.ndarray | None:
        """
        Return normalized angular-momentum directions for hierarchical pairings.

        The default implementation reuses
        `get_normalized_angular_momentum(...)`.

        Args:
            stars:  Unresolved stellar or system particles.
            primary_index: Indices of existing multiple systems.
            companion_index: Indices of selected companions.
            hierarchy: Current <a href="#dcaf.factory.multiplicity--hierarchy-format">hierarchy dictionaries</a>.

        Returns:
            (numpy.ndarray | None): Outer-orbit unit angular-
                momentum vectors, or `None` for random orientations.
        """
        return self.get_normalized_angular_momentum(
            stars, primary_index, companion_index, **kwargs
        )

    def select_binaries(
        self,
        stars: Particles,
        nbinaries: int,
    ) -> tuple[list[int], list[int]]:
        """Select primary and companion indices from `stars`.

        Args:
            stars: Input stellar particles.
            nbinaries: Number of binary pairs to select.

        Returns:
            (tuple[list[int], list[int]]): Primary and companion index lists.
        """
        equal_mass = numpy.std(stars.mass.value_in(units.MSun)) == 0.0
        mass = stars.mass 

        primary_index = []
        companion_index = []
        available_indexes = list(range(len(mass)))
        mmin = mass.min()

        for _ in range(nbinaries):
            if len(available_indexes) > 2:
                mindex = mmin
                if not equal_mass:
                    while mindex == mmin:
                        index = numpy.random.choice(available_indexes)
                        mindex = mass[index]
                        if len(available_indexes) == 1:
                            break
                else:
                    index = numpy.random.choice(available_indexes)
                    mindex = mass[index]

                primary_index.append(index)
                available_indexes.remove(index)

                m2 = self.choose_companion(mass[index], mass[available_indexes], equal_mass)
                index2 = index
                while index2 not in available_indexes:
                    candidates = numpy.where(mass == m2)[0]
                    index2 = numpy.random.choice(candidates)
                    mindex = mass[index2]

                companion_index.append(index2)
                available_indexes.remove(index2)

                if mindex == mmin and len(available_indexes) > 0:
                    mmin = mass[available_indexes].min()
            else:
                if len(available_indexes) == 2:
                    index1 = available_indexes[0]
                    index2 = available_indexes[1]
                    if mass[index1] >= mass[index2]:
                        primary_index.append(index1)
                        companion_index.append(index2)
                    else:
                        primary_index.append(index2)
                        companion_index.append(index1)
                break

        return primary_index, companion_index

    def apply_population(self, stars: Particles) -> dict:
        """
        Build the configured multiplicity population internally from a clean
        `Particles`.

        This is a convenience wrapper for the standard staged workflow. The
        current default implementation supports `[single, binary, triple]`
        population fractions and constructs triples by reprocessing the
        unresolved systems after the first binary pass.

        The returned `hierarchy` is a list of dictionaries that follows the
        <a href="#dcaf.factory.multiplicity--hierarchy-format">hierarchy format</a>
        described above.
        Each entry corresponds to the particle at the same index in
        `unresolved_stars`.

        Args:
            stars: Input stellar particles.

        Returns:
            (dict): Resolved and unresolved particles, hierarchy,
                orbital properties, and achieved population fractions.

        Raises:
            ValueError: If `population_fraction` is not configured.
            NotImplementedError: If a nonzero population above triples is
                requested.
        """
        if self.population_fraction is None:
            raise ValueError(
                "apply_population(...) requires population_fraction to be configured."
            )

        fractions = numpy.array(self.population_fraction, dtype=float)
        fractions = fractions / fractions.sum()

        if len(fractions) > 3 and numpy.any(fractions[3:] > 0.0):
            raise NotImplementedError(
                "apply_population(...) currently supports only singles, binaries, and triples."
            )

        mean_multiplicity = numpy.sum(
            fractions * numpy.arange(1, len(fractions) + 1, dtype=float)
        )
        nsystems_guess = float(len(stars)) / mean_multiplicity

        counts = numpy.zeros(len(fractions), dtype=int)
        if len(fractions) > 1:
            counts[1:] = numpy.rint(fractions[1:] * nsystems_guess).astype(int)

        remaining_stars = len(stars) - numpy.sum(
            counts[1:] * numpy.arange(2, len(fractions) + 1, dtype=int)
        )
        while remaining_stars < 0:
            for i in range(len(counts) - 1, 0, -1):
                if counts[i] > 0:
                    counts[i] -= 1
                    remaining_stars += i + 1
                    break
        counts[0] = remaining_stars

        first_pass_nbinaries = int(numpy.sum(counts[1:]))
        result = self.apply(stars, force_n_binaries=first_pass_nbinaries)

        if len(counts) > 2 and counts[2] > 0:
            result = self.apply(
                result["unresolved_stars"],
                hierarchy=result["hierarchy"],
                force_n_binaries=int(numpy.sum(counts[2:])),
            )

        return result

    def make_binaries(
        self,
        stars: Particles,
        hierarchy: list[dict] = None,
        force_n_binaries: int = None,
    ) -> tuple[dict, dict]:
        """
        Resolve binary components around the centre-of-mass phase-space points
        of the selected primary stars, using the current class configuration.

        Args:
            stars: Stellar or unresolved-system particles to pair.
            hierarchy: Existing <a href="#dcaf.factory.multiplicity--hierarchy-format">hierarchy dictionaries</a> for
                recursive higher-order pairing.
            force_n_binaries: Binary count for this pairing pass.

        Returns:
            (tuple[dict, dict]): Orbital and hierarchy data, plus internal
                resolved-component and index data.
        """
        mass = stars.mass
        if force_n_binaries is None:
            nbinaries = self.get_number_of_binaries(stars)
        else:
            nbinaries = int(force_n_binaries)
        if hierarchy is None:
            primary_index, companion_index = self.select_binaries(stars, nbinaries)
        else:
            primary_index = []
            companion_index = []
            available_indexes = list(range(len(stars)))

            for _ in range(nbinaries):
                multiple_indexes = [
                    i for i in available_indexes
                    if len(hierarchy[i]["members"]) > 1
                ]
                if len(multiple_indexes) == 0:
                    break

                index = numpy.random.choice(multiple_indexes)
                index2 = self.select_higher_order_companion(
                    index, stars, hierarchy, available_indexes
                )
                if index2 is None:
                    available_indexes.remove(index)
                    continue

                primary_index.append(index)
                companion_index.append(index2)
                available_indexes.remove(index)
                available_indexes.remove(index2)

        binary_index = list(primary_index)
        used = set(primary_index) | set(companion_index)
        single_index = [i for i in range(len(stars)) if i not in used]

        max_semi_major_axis = self._resolve_max_semi_major_axis(stars)

        nbinaries = len(primary_index)

        if nbinaries == 0:
            resolved_stars = stars.copy()
            resolved_stars.system_id = numpy.arange(1, len(resolved_stars) + 1, dtype=int)
            hierarchy = []
            for i in range(len(resolved_stars)):
                hierarchy.append(
                    dict(
                        members=[i],
                        periods=[],
                        semi_major_axes=[],
                        eccentricities=[],
                        lhat=[],
                        _resolved_particles=resolved_stars[i:i+1].copy(),
                    )
                )
            return (
                {
                    "resolved_stars": resolved_stars,
                    "unresolved_stars": resolved_stars.copy(),
                    "hierarchy": hierarchy,
                    "periods": [] | units.day,
                    "semi_major_axes": [] | units.AU,
                    "eccentricities": numpy.array([]),
                    "nbinaries": 0,
                },
                {
                    "primary_components": Particles(),
                    "secondary_components": Particles(),
                    "single_particles": resolved_stars.copy(),
                    "binary_systems": Particles(),
                    "primary_index": [],
                    "companion_index": [],
                    "single_index": list(range(len(stars))),
                },
            )

        primary_index = list(primary_index)
        companion_index = list(companion_index)
        binary_index = list(binary_index)
        single_index = list(single_index)

        binary_mass = mass[primary_index] + mass[companion_index]
        if hierarchy is None:
            periods = self.get_periods(
                stars,
                primary_index,
                companion_index,
                max_semi_major_axis=max_semi_major_axis,
            )
            ecc = self.get_eccentricities(stars, primary_index, companion_index)
            lhat = self.get_normalized_angular_momentum(
                stars,
                primary_index,
                companion_index,
                max_semi_major_axis=max_semi_major_axis,
            )
        else:
            periods = self.get_higher_order_periods(
                stars,
                primary_index,
                companion_index,
                hierarchy,
                max_semi_major_axis=max_semi_major_axis,
            )
            ecc = self.get_higher_order_eccentricities(
                stars,
                primary_index,
                companion_index,
                hierarchy,
                max_semi_major_axis=max_semi_major_axis,
            )
            lhat = self.get_higher_order_normalized_angular_momentum(
                stars,
                primary_index,
                companion_index,
                hierarchy,
                max_semi_major_axis=max_semi_major_axis,
            )

        semi_major_axes = (
            (periods.value_in(units.yr) ** 2) * binary_mass.value_in(units.MSun)
        ) ** (1.0 / 3.0) | units.AU

        vc = (G * binary_mass * (1.0 - ecc) / semi_major_axes / (1.0 + ecc)).sqrt()
        separation = semi_major_axes * (1.0 + ecc)

        pi_angle = numpy.random.uniform(high=2.0 * numpy.pi, size=nbinaries)
        omega = numpy.random.uniform(high=2.0 * numpy.pi, size=nbinaries)
        zi = numpy.random.uniform(high=numpy.pi, size=nbinaries)

        if lhat is not None:
            lhat = numpy.asarray(lhat, dtype=float)
            if lhat.shape != (nbinaries, 3):
                raise ValueError(
                    f"lhat must have shape ({nbinaries}, 3), got {lhat.shape}."
                )

            valid = numpy.all(numpy.isfinite(lhat), axis=1)
            if numpy.any(valid):
                norms = numpy.linalg.norm(lhat[valid], axis=1)
                valid_indices = numpy.where(valid)[0]
                nonzero = norms > 0.0
                if numpy.any(nonzero):
                    use = valid_indices[nonzero]
                    lhat[use] = lhat[use] / norms[nonzero][:, None]

                    lz = numpy.clip(lhat[use, 2], -1.0, 1.0)
                    zi[use] = numpy.arccos(lz)
                    omega[use] = numpy.arctan2(lhat[use, 0], -lhat[use, 1])

        px1 = numpy.cos(pi_angle) * numpy.cos(omega) - numpy.sin(pi_angle) * numpy.sin(omega) * numpy.cos(zi)
        qx1 = -numpy.sin(pi_angle) * numpy.cos(omega) - numpy.cos(pi_angle) * numpy.sin(omega) * numpy.cos(zi)
        px2 = numpy.cos(pi_angle) * numpy.sin(omega) + numpy.sin(pi_angle) * numpy.cos(omega) * numpy.cos(zi)
        qx2 = -numpy.sin(pi_angle) * numpy.sin(omega) + numpy.cos(pi_angle) * numpy.cos(omega) * numpy.cos(zi)
        px3 = numpy.sin(pi_angle) * numpy.sin(zi)
        qx3 = numpy.cos(pi_angle) * numpy.sin(zi)

        lhat = numpy.column_stack((
            numpy.sin(omega) * numpy.sin(zi),
            -numpy.cos(omega) * numpy.sin(zi),
            numpy.cos(zi),
        ))

        xrel = px1 * separation
        yrel = px2 * separation
        zrel = px3 * separation
        vxrel = qx1 * vc
        vyrel = qx2 * vc
        vzrel = qx3 * vc

        binary_systems = Particles(nbinaries)
        binary_systems.mass = binary_mass
        binary_systems.position = stars[binary_index].position
        binary_systems.velocity = stars[binary_index].velocity
        binary_systems.m1 = mass[primary_index]
        binary_systems.m2 = mass[companion_index]
        binary_systems.binary = True

        primary_components = Particles(nbinaries)
        secondary_components = Particles(nbinaries)

        primary_components.mass = mass[primary_index]
        secondary_components.mass = mass[companion_index]
        primary_components.radius = stars[binary_index].radius
        secondary_components.radius = stars[binary_index].radius

        primary_components.x = binary_systems.x + secondary_components.mass * xrel / binary_systems.mass
        primary_components.y = binary_systems.y + secondary_components.mass * yrel / binary_systems.mass
        primary_components.z = binary_systems.z + secondary_components.mass * zrel / binary_systems.mass

        secondary_components.x = primary_components.x - xrel
        secondary_components.y = primary_components.y - yrel
        secondary_components.z = primary_components.z - zrel

        primary_components.vx = binary_systems.vx + secondary_components.mass * vxrel / binary_systems.mass
        primary_components.vy = binary_systems.vy + secondary_components.mass * vyrel / binary_systems.mass
        primary_components.vz = binary_systems.vz + secondary_components.mass * vzrel / binary_systems.mass

        secondary_components.vx = primary_components.vx - vxrel
        secondary_components.vy = primary_components.vy - vyrel
        secondary_components.vz = primary_components.vz - vzrel

        primary_components.in_binary = True
        secondary_components.in_binary = True

        single_particles = stars[single_index].copy()
        if len(single_particles) > 0:
            single_particles.binary = False
            single_particles.in_binary = False

        hierarchy = []
        for i in range(nbinaries):
            hierarchy.append(
                dict(
                    members=[primary_index[i], companion_index[i]],
                    periods=[periods[i]],
                    semi_major_axes=[semi_major_axes[i]],
                    eccentricities=[ecc[i]],
                    lhat=[lhat[i].copy()],
                )
            )
        for idx in single_index:
            hierarchy.append(
                dict(
                    members=[idx],
                    periods=[],
                    semi_major_axes=[],
                    eccentricities=[],
                    lhat=[],
                )
            )

        return (
            {
                "hierarchy": hierarchy,
                "periods": periods,
                "semi_major_axes": semi_major_axes,
                "eccentricities": ecc,
                "lhat": lhat,
                "nbinaries": nbinaries,
            },
            {
                "primary_components": primary_components,
                "secondary_components": secondary_components,
                "single_particles": single_particles,
                "binary_systems": binary_systems,
                "primary_index": primary_index,
                "companion_index": companion_index,
                "single_index": single_index,
            },
        )

    def apply(
        self,
        stars: Particles,
        hierarchy: list[dict] = None,
        force_n_binaries: int = None,
    ) -> dict:
        """
        Build the configured primordial-binary population on top of an existing
        `Particles` and return both the resolved stars and the unresolved
        system-level particle set.

        If `force_n_binaries` is provided, it overrides the currently
        configured binary count for this call only. This is mainly intended for
        manual staged construction and for internal orchestration from
        `apply_population(...)`.


        Args:
            stars: Input stellar or unresolved-system particles.
            hierarchy: Existing <a href="#dcaf.factory.multiplicity--hierarchy-format">hierarchy dictionaries</a> for
                a recursive pairing pass.
            force_n_binaries: Binary count for this call.

        Returns:
            (dict): `resolved_stars`, `unresolved_stars`, hierarchy, orbital
                properties, and achieved population fractions.

        Raises:
            NotImplementedError: If triples are requested through `apply` rather
                than `apply_population`.
        """
        if (
            hierarchy is None
            and force_n_binaries is None
            and self.nbinaries is None
            and self.population_fraction is not None
            and len(self.population_fraction) > 2
            and numpy.any(numpy.array(self.population_fraction[2:]) > 0.0)
        ):
            raise NotImplementedError(
                "Use apply_population(...) to match population_fraction with triples."
            )

        hierarchy_in = hierarchy

        if hierarchy_in is None:
            stars = self.preprocess_stars(stars)


        data, internal = self.make_binaries(
            stars,
            hierarchy=hierarchy_in,
            force_n_binaries=force_n_binaries,
        )

        nbinaries = data["nbinaries"]

        if nbinaries > 0 or hierarchy_in is not None:
            ordered_systems = []
            for i in range(nbinaries):
                ordered_systems.append(
                    (
                        min(internal["primary_index"][i], internal["companion_index"][i]),
                        "binary",
                        i,
                    )
                )
            for i, idx in enumerate(internal["single_index"]):
                ordered_systems.append((idx, "single", i))
            ordered_systems.sort(key=lambda item: item[0])

            resolved_stars = Particles()
            unresolved_stars = Particles()
            hierarchy = []
            system_id = 1

            for _, kind, i in ordered_systems:
                if kind == "binary":
                    primary = internal["primary_components"][i].copy()
                    secondary = internal["secondary_components"][i].copy()
                    binary_system = internal["binary_systems"][i].copy()

                    if hierarchy_in is None:
                        primary_resolved = Particles()
                        secondary_resolved = Particles()
                        primary_resolved.add_particle(primary.copy())
                        secondary_resolved.add_particle(secondary.copy())
                    else:
                        primary_entry = hierarchy_in[internal["primary_index"][i]]
                        secondary_entry = hierarchy_in[internal["companion_index"][i]]

                        primary_resolved = primary_entry["_resolved_particles"].copy()
                        secondary_resolved = secondary_entry["_resolved_particles"].copy()

                        if self.higher_order_mode == "hierarchical":
                            for target, resolved_subset in (
                                (primary, primary_resolved),
                                (secondary, secondary_resolved),
                            ):
                                old_com = resolved_subset.center_of_mass()
                                old_com_velocity = (
                                    resolved_subset.center_of_mass_velocity()
                                )

                                dx = resolved_subset.x - old_com.x
                                dy = resolved_subset.y - old_com.y
                                dz = resolved_subset.z - old_com.z
                                dvx = resolved_subset.vx - old_com_velocity.x
                                dvy = resolved_subset.vy - old_com_velocity.y
                                dvz = resolved_subset.vz - old_com_velocity.z

                                resolved_subset.x = target.x + dx
                                resolved_subset.y = target.y + dy
                                resolved_subset.z = target.z + dz
                                resolved_subset.vx = target.vx + dvx
                                resolved_subset.vy = target.vy + dvy
                                resolved_subset.vz = target.vz + dvz
                        else:
                            old_primary_com = primary_resolved.center_of_mass()
                            old_primary_com_velocity = (
                                primary_resolved.center_of_mass_velocity()
                            )

                            primary_anchor = primary_resolved[0]
                            primary_offset_x = primary_anchor.x - old_primary_com.x
                            primary_offset_y = primary_anchor.y - old_primary_com.y
                            primary_offset_z = primary_anchor.z - old_primary_com.z
                            primary_offset_vx = (
                                primary_anchor.vx - old_primary_com_velocity.x
                            )
                            primary_offset_vy = (
                                primary_anchor.vy - old_primary_com_velocity.y
                            )
                            primary_offset_vz = (
                                primary_anchor.vz - old_primary_com_velocity.z
                            )

                            target_com = binary_system.copy()
                            total_mass = binary_system.mass
                            companion_mass = secondary_resolved.mass.sum()
                            primary_mass = primary_resolved.mass.sum()

                            anchor_x = (
                                target_com.x
                                + (
                                    primary_mass * primary_offset_x
                                    - companion_mass * (secondary.x - primary.x)
                                )
                                / total_mass
                            )
                            anchor_y = (
                                target_com.y
                                + (
                                    primary_mass * primary_offset_y
                                    - companion_mass * (secondary.y - primary.y)
                                )
                                / total_mass
                            )
                            anchor_z = (
                                target_com.z
                                + (
                                    primary_mass * primary_offset_z
                                    - companion_mass * (secondary.z - primary.z)
                                )
                                / total_mass
                            )
                            anchor_vx = (
                                target_com.vx
                                + (
                                    primary_mass * primary_offset_vx
                                    - companion_mass * (secondary.vx - primary.vx)
                                )
                                / total_mass
                            )
                            anchor_vy = (
                                target_com.vy
                                + (
                                    primary_mass * primary_offset_vy
                                    - companion_mass * (secondary.vy - primary.vy)
                                )
                                / total_mass
                            )
                            anchor_vz = (
                                target_com.vz
                                + (
                                    primary_mass * primary_offset_vz
                                    - companion_mass * (secondary.vz - primary.vz)
                                )
                                / total_mass
                            )

                            dx = primary_resolved.x - primary_anchor.x
                            dy = primary_resolved.y - primary_anchor.y
                            dz = primary_resolved.z - primary_anchor.z
                            dvx = primary_resolved.vx - primary_anchor.vx
                            dvy = primary_resolved.vy - primary_anchor.vy
                            dvz = primary_resolved.vz - primary_anchor.vz

                            primary_resolved.x = anchor_x + dx
                            primary_resolved.y = anchor_y + dy
                            primary_resolved.z = anchor_z + dz
                            primary_resolved.vx = anchor_vx + dvx
                            primary_resolved.vy = anchor_vy + dvy
                            primary_resolved.vz = anchor_vz + dvz

                            companion_anchor = secondary_resolved[0]
                            cdx = secondary_resolved.x - companion_anchor.x
                            cdy = secondary_resolved.y - companion_anchor.y
                            cdz = secondary_resolved.z - companion_anchor.z
                            cdvx = secondary_resolved.vx - companion_anchor.vx
                            cdvy = secondary_resolved.vy - companion_anchor.vy
                            cdvz = secondary_resolved.vz - companion_anchor.vz

                            secondary_resolved.x = primary_resolved[0].x + (
                                secondary.x - primary.x
                            ) + cdx
                            secondary_resolved.y = primary_resolved[0].y + (
                                secondary.y - primary.y
                            ) + cdy
                            secondary_resolved.z = primary_resolved[0].z + (
                                secondary.z - primary.z
                            ) + cdz
                            secondary_resolved.vx = primary_resolved[0].vx + (
                                secondary.vx - primary.vx
                            ) + cdvx
                            secondary_resolved.vy = primary_resolved[0].vy + (
                                secondary.vy - primary.vy
                            ) + cdvy
                            secondary_resolved.vz = primary_resolved[0].vz + (
                                secondary.vz - primary.vz
                            ) + cdvz

                    primary_resolved.system_id = system_id
                    secondary_resolved.system_id = system_id
                    binary_system.system_id = system_id

                    resolved_stars.add_particles(primary_resolved)
                    resolved_stars.add_particles(secondary_resolved)
                    unresolved_stars.add_particle(binary_system)

                    merged_members = list(
                        range(
                            len(resolved_stars) - len(primary_resolved) - len(secondary_resolved),
                            len(resolved_stars),
                        )
                    )
                    merged_resolved = Particles()
                    merged_resolved.add_particles(primary_resolved.copy())
                    merged_resolved.add_particles(secondary_resolved.copy())

                    if hierarchy_in is None:
                        merged_periods = [data["periods"][i]]
                        merged_semi_major_axes = [data["semi_major_axes"][i]]
                        merged_eccentricities = [data["eccentricities"][i]]
                        merged_lhat = [data["lhat"][i].copy()]
                    else:
                        primary_entry = hierarchy_in[internal["primary_index"][i]]
                        secondary_entry = hierarchy_in[internal["companion_index"][i]]
                        merged_periods = (
                            list(primary_entry["periods"])
                            + list(secondary_entry["periods"])
                            + [data["periods"][i]]
                        )
                        merged_semi_major_axes = (
                            list(primary_entry["semi_major_axes"])
                            + list(secondary_entry["semi_major_axes"])
                            + [data["semi_major_axes"][i]]
                        )
                        merged_eccentricities = (
                            list(primary_entry["eccentricities"])
                            + list(secondary_entry["eccentricities"])
                            + [data["eccentricities"][i]]
                        )
                        merged_lhat = (
                            [numpy.array(v, copy=True) for v in primary_entry["lhat"]]
                            + [numpy.array(v, copy=True) for v in secondary_entry["lhat"]]
                            + [data["lhat"][i].copy()]
                        )

                    hierarchy.append(
                        dict(
                            members=merged_members,
                            periods=merged_periods,
                            semi_major_axes=merged_semi_major_axes,
                            eccentricities=merged_eccentricities,
                            lhat=merged_lhat,
                            _resolved_particles=merged_resolved,
                        )
                    )
                else:
                    single = internal["single_particles"][i].copy()
                    unresolved_single = Particles()
                    unresolved_single.add_particle(single.copy())
                    unresolved_single.system_id = system_id

                    if hierarchy_in is None:
                        resolved_single = Particles()
                        resolved_single.add_particle(single.copy())
                    else:
                        entry = hierarchy_in[internal["single_index"][i]]
                        resolved_single = entry["_resolved_particles"].copy()
                        old_com = resolved_single.center_of_mass()
                        old_com_velocity = resolved_single.center_of_mass_velocity()

                        dx = resolved_single.x - old_com.x
                        dy = resolved_single.y - old_com.y
                        dz = resolved_single.z - old_com.z
                        dvx = resolved_single.vx - old_com_velocity.x
                        dvy = resolved_single.vy - old_com_velocity.y
                        dvz = resolved_single.vz - old_com_velocity.z

                        target = unresolved_single[0]
                        resolved_single.x = target.x + dx
                        resolved_single.y = target.y + dy
                        resolved_single.z = target.z + dz
                        resolved_single.vx = target.vx + dvx
                        resolved_single.vy = target.vy + dvy
                        resolved_single.vz = target.vz + dvz

                    resolved_single.system_id = system_id

                    resolved_stars.add_particles(resolved_single)
                    unresolved_stars.add_particle(unresolved_single[0].copy())

                    if hierarchy_in is None:
                        periods_single = []
                        semi_major_axes_single = []
                        eccentricities_single = []
                        lhat_single = []
                    else:
                        entry = hierarchy_in[internal["single_index"][i]]
                        periods_single = list(entry["periods"])
                        semi_major_axes_single = list(entry["semi_major_axes"])
                        eccentricities_single = list(entry["eccentricities"])
                        lhat_single = [numpy.array(v, copy=True) for v in entry["lhat"]]

                    hierarchy.append(
                        dict(
                            members=list(
                                range(
                                    len(resolved_stars) - len(resolved_single),
                                    len(resolved_stars),
                                )
                            ),
                            periods=periods_single,
                            semi_major_axes=semi_major_axes_single,
                            eccentricities=eccentricities_single,
                            lhat=lhat_single,
                            _resolved_particles=resolved_single.copy(),
                        )
                    )

                system_id += 1

            data["resolved_stars"] = resolved_stars
            data["unresolved_stars"] = unresolved_stars
            data["hierarchy"] = hierarchy

        unresolved_stars = data["unresolved_stars"]
        max_order = 2
        if "hierarchy" in data and len(data["hierarchy"]) > 0:
            max_order = max(
                max_order,
                max(len(entry["members"]) for entry in data["hierarchy"]),
            )
        if self.population_fraction is not None:
            max_order = max(max_order, len(self.population_fraction))

        population_fraction = [0.0] * max_order
        if len(unresolved_stars) > 0 and "hierarchy" in data:
            for entry in data["hierarchy"]:
                order = len(entry["members"])
                population_fraction[order - 1] += 1.0
            population_fraction = [
                value / len(unresolved_stars) for value in population_fraction
            ]

        data["nbinaries"] = nbinaries
        data["population_fraction"] = population_fraction
        data["binary_fraction"] = population_fraction[1]

        return data



class FieldBinaryPopulation(BinaryPopulation):
    """
    This class implements \
    [Cournoyer-Cloutier et al. (2024)](https://iopscience.iop.org/article/10.3847/1538-4357/ad90b3) \
    binary population generation, which aims to reproduce observational\
    statistics obtained by Moe & Di Stefano (2017), Winters et al. (2019) and\
    Offner et al. (2022).

    This is an example of how we can implement a more complex population using
    the `BinaryPopulation` class.

    Notes: 
    - The apply function here works differently than the standard D-CAF. The
      provided stars will not be paired between them. Rather, companions will be
      created for primaries the algorithm decides to become a binary.
      However, the total mass will be respected (within small variations caused
      by sampling) by trimming the excess of stars. This trimming is random, and
      by system.
    - This class do not requires binary fraction. The binary fraction is
      dependent on mass and is decided by the algorithm.

    Args:
        mult_frac:  Multiplicity-fraction prescription passed to
            `field_binary_population.get_multiplicity`.
        pdist: Period-distribution prescription.
        qdist: Mass-ratio-distribution prescription.
        edist: Eccentricity-distribution prescription.
        min_mass: Minimum companion mass in solar masses.
        max_semi_major_axis: Optional maximum binary semimajor axis. Unlike
            `BinaryPopulation`, this cap is enforced only when explicitly set
            during population preprocessing.

    Attributes:
        mult_frac (str): Multiplicity-fraction prescription.
        pdist (str): Period-distribution prescription.
        qdist (str): Mass-ratio-distribution prescription.
        edist (str): Eccentricity-distribution prescription.
        min_mass (float): Minimum companion mass in solar masses.
        max_semi_major_axis (Quantity): Optional maximum binary semimajor axis.
    """

    def __init__(
        self,
        mult_frac: str = "field",
        pdist: str = "inner",
        qdist: str = "field",
        edist: str = "field",
        min_mass: float = 0.08,
        max_semi_major_axis: Quantity = None,
        **kwargs: object,
    ) -> None:
        self.mult_frac = mult_frac
        self.pdist = pdist
        self.qdist = qdist
        self.edist = edist
        self.min_mass = min_mass
        self._enforce_max_semi_major_axis = max_semi_major_axis is not None

        super().__init__(
            nbinaries=0,
            max_semi_major_axis=max_semi_major_axis,
            **kwargs,
        )

    def preprocess_stars(self, stars: Particles) -> Particles:
        """
        On this version of the class. This function assign companions and binary
        properties to each star. It create the companion stars and append the
        necessary binary properties. 
        functions called later will just read what we decide here.

        Args:
            stars: Input stellar particles.

        Returns:
            (Particles): Copied and augmented particles, with
                created companions and binary bookkeeping attributes.

        Raises:
            ValueError: If no whole system fits within the input mass budget.
            ValueError: If `max_semi_major_axis` is non-positive, cannot admit
                an orbit from the field distributions, or cannot be sampled
                within 10000 attempts.
        """
        stars = stars.copy()
        target_mass = stars.mass.sum()

        if hasattr(stars, "secondary_id"):
            return stars

        masses = stars.mass.value_in(units.MSun)

        primary_mask, _ = fbp.get_multiplicity(
            masses,
            mult_frac=self.mult_frac,
        )
        primary_index = np.where(primary_mask)[0]
        primary_masses = masses[primary_index]

        self.nbinaries = len(primary_index)

        secondary_id = -np.ones(len(stars), dtype=int)
        is_secondary = np.zeros(len(stars), dtype=bool)
        binary_period = np.zeros(len(stars)) | units.day

        stars.secondary_id = secondary_id
        stars.is_secondary = is_secondary
        stars.binary_period = binary_period

        if len(primary_index) == 0:
            return stars

        periods = fbp.get_periods(primary_masses, pdist=self.pdist) | units.day
        q = fbp.get_mass_ratios(
            primary_masses,
            periods.value_in(units.day),
            qdist=self.qdist,
            mmin=self.min_mass,
        )

        if self._enforce_max_semi_major_axis:
            max_semi_major_axis = self.max_semi_major_axis
            if max_semi_major_axis <= 0 | max_semi_major_axis.unit:
                raise ValueError("max_semi_major_axis must be positive.")

            minimum_period = 10.0 ** 0.2 | units.day
            minimum_companion_mass = np.maximum(
                0.1 * primary_masses,
                self.min_mass,
            )
            minimum_semi_major_axis = (
                (minimum_period.value_in(units.yr) ** 2)
                * (primary_masses + minimum_companion_mass)
            ) ** (1.0 / 3.0) | units.AU
            if np.any(minimum_semi_major_axis > max_semi_major_axis):
                raise ValueError(
                    "max_semi_major_axis is smaller than an orbit allowed by "
                    "the field period and mass-ratio distributions."
                )

            semi_major_axes = (
                (periods.value_in(units.yr) ** 2)
                * (primary_masses + primary_masses * q)
            ) ** (1.0 / 3.0) | units.AU
            pending = np.where(semi_major_axes > max_semi_major_axis)[0]
            attempts = 0
            while len(pending) > 0:
                attempts += 1
                if attempts > 10000:
                    raise ValueError(
                        "Unable to sample field binaries within "
                        "max_semi_major_axis after 10000 attempts."
                    )

                candidate_periods = (
                    fbp.get_periods(primary_masses[pending], pdist=self.pdist)
                    | units.day
                )
                candidate_q = fbp.get_mass_ratios(
                    primary_masses[pending],
                    candidate_periods.value_in(units.day),
                    qdist=self.qdist,
                    mmin=self.min_mass,
                )
                candidate_semi_major_axes = (
                    (candidate_periods.value_in(units.yr) ** 2)
                    * (
                        primary_masses[pending]
                        + primary_masses[pending] * candidate_q
                    )
                ) ** (1.0 / 3.0) | units.AU
                accepted = candidate_semi_major_axes <= max_semi_major_axis

                accepted_indices = pending[accepted]
                periods[accepted_indices] = candidate_periods[accepted]
                q[accepted_indices] = candidate_q[accepted]
                pending = pending[~accepted]

        companion_masses = primary_masses * q
        secondaries = Particles(len(primary_index))
        secondaries.mass = companion_masses | units.MSun
        secondaries.secondary_id = -np.ones(len(secondaries), dtype=int)
        secondaries.is_secondary = np.ones(len(secondaries), dtype=bool)
        secondaries.binary_period = np.zeros(len(secondaries)) | units.day

        if hasattr(stars, "radius"):
            secondaries.radius = stars[primary_index].radius
        if hasattr(stars, "x"):
            secondaries.position = stars[primary_index].position
        if hasattr(stars, "vx"):
            secondaries.velocity = stars[primary_index].velocity

        n_old = len(stars)
        stars.add_particles(secondaries)

        companion_index = np.arange(n_old, n_old + len(secondaries))
        stars.secondary_id[primary_index] = companion_index
        stars.binary_period[primary_index] = periods
        stars.is_secondary[companion_index] = True

        # Trim the excess of systems so that we have the desired total mass

        if stars.mass.sum() > target_mass:
            systems = []

            # Keep complete systems, so binaries are never split by trimming.
            for i in range(len(stars)):
                if stars.is_secondary[i]:
                    continue

                j = int(stars.secondary_id[i])
                members = [i] if j < 0 else [i, j]
                system_mass = stars[members].mass.sum()
                systems.append((members, system_mass))

            selected_systems = []
            cumulative_mass = 0 | target_mass.unit
            for i in np.random.permutation(len(systems)):
                members, system_mass = systems[i]
                if cumulative_mass + system_mass > target_mass:
                    continue

                selected_systems.append(members)
                cumulative_mass += system_mass

            if len(selected_systems) == 0:
                raise ValueError(
                    "No system fits within the target mass budget; "
                    "cannot trim with a <= whole-system rule."
                )

            # Preserve a stable particle order after random system selection.
            selected_systems.sort(key=lambda members: members[0])
            systems = selected_systems
            kept_indices = [idx for members in systems for idx in members]
            trimmed = stars[kept_indices].copy()

            # Rebuild binary bookkeeping because particle indices changed.
            old_to_new = {old: new for new, old in enumerate(kept_indices)}
            secondary_id = -np.ones(len(trimmed), dtype=int)
            is_secondary = np.zeros(len(trimmed), dtype=bool)
            binary_period = np.zeros(len(trimmed)) | units.day

            for members in systems:
                if len(members) == 2:
                    old_primary, old_secondary = members
                    new_primary = old_to_new[old_primary]
                    new_secondary = old_to_new[old_secondary]

                    secondary_id[new_primary] = new_secondary
                    is_secondary[new_secondary] = True
                    binary_period[new_primary] = stars.binary_period[old_primary]

            trimmed.secondary_id = secondary_id
            trimmed.is_secondary = is_secondary
            trimmed.binary_period = binary_period

            stars = trimmed

        self.nbinaries = int(np.sum(stars.secondary_id >= 0))

        return stars

    def select_binaries(
        self,
        stars: Particles,
        nbinaries: int,
    ) -> tuple[list[int], list[int]]:
        """
        Return the indexes of primaries and secondaries.
        On this FieldBinaryPopulation class, that was decided in the
        preproecess_stars function. Here we just hand it over.

        Args:
            stars: Prepared stellar particles with `secondary_id`.
            nbinaries: Expected number of binary systems.

        Returns:
            (tuple[list[int], list[int]]): Primary and companion
                index lists.

        Raises:
            ValueError: If prepared binary count differs from the
                requested count.
        """
        primary_index = np.where(stars.secondary_id >= 0)[0].tolist()
        companion_index = stars.secondary_id[primary_index].astype(int).tolist()

        # a sanity check
        if len(primary_index) != nbinaries:
            raise ValueError(
                f"Prepared {len(primary_index)} binaries but DCAF requested {nbinaries}."
            )

        return primary_index, companion_index

    def get_periods(
        self,
        stars: Particles,
        primary_index: list[int],
        companion_index: list[int],
        sample: bool = False,
        **kwargs: Quantity,
    ) -> Quantity:
        """Return stored or newly sampled field-binary periods.

        Args:
            stars: Prepared stellar particles.
            primary_index: Indices of primary stars.
            companion_index: Indices of companion stars.
            sample: Whether to resample periods instead of using
                `binary_period`.

        Returns:
            (Quantity): Orbital periods in days.
        """
        m = stars.mass[primary_index].value_in(units.MSun)
        if not sample:
            return stars[primary_index].binary_period

        return fbp.get_periods(m, pdist=self.pdist) | units.day

    def get_eccentricities(
        self,
        stars: Particles,
        primary_index: list[int],
        companion_index: list[int],
        **kwargs: Quantity,
    ) -> numpy.ndarray:
        """Sample eccentricities from the field prescription.

        Args:
            stars: Prepared stellar particles.
            primary_index: Indices of primary stars.
            companion_index: Indices of companion stars.
            **kwargs: Additional ignored pairing context.

        Returns:
            (numpy.ndarray): One eccentricity per binary.
        """
        m = stars.mass[primary_index].value_in(units.MSun)
        p = stars.binary_period[primary_index].value_in(units.day)

        ecc = fbp.get_eccentricities(m,p, edist = self.edist )

        return ecc

def synchronize_resolved_with_unresolved(
    unresolved_stars: Particles,
    resolved_stars: Particles,
) -> Particles:
    """
    Move resolved components so each system matches the given unresolved COM.

    Args:
        unresolved_stars: System centre-of-mass particles with `system_id`
            values.
        resolved_stars: Resolved components with matching
            `system_id` values.

    Returns:
        (Particles): Copy of `resolved_stars` translated to the corresponding
            unresolved-system centres of mass.
    """
    if not hasattr(unresolved_stars, "system_id"):
        raise AttributeError("unresolved_stars must have a system_id attribute.")
    if not hasattr(resolved_stars, "system_id"):
        raise AttributeError("resolved_stars must have a system_id attribute.")

    updated = resolved_stars.copy()

    unresolved_ids = list(unresolved_stars.system_id)
    resolved_ids = list(updated.system_id)

    for system_id in unresolved_ids:
        unresolved_mask = [sid == system_id for sid in unresolved_ids]
        resolved_mask = [sid == system_id for sid in resolved_ids]

        unresolved_system = unresolved_stars[unresolved_mask]
        resolved_system = updated[resolved_mask]

        if len(unresolved_system) != 1:
            raise ValueError(
                "Each system_id must appear exactly once in unresolved_stars."
            )
        if len(resolved_system) == 0:
            raise ValueError(
                f"system_id {system_id} is present in unresolved_stars but missing in resolved_stars."
            )

        target = unresolved_system[0]

        if len(resolved_system) == 1:
            resolved_system.position = target.position
            resolved_system.velocity = target.velocity
            continue

        old_com = resolved_system.center_of_mass()
        old_com_velocity = resolved_system.center_of_mass_velocity()

        dx = resolved_system.x - old_com.x
        dy = resolved_system.y - old_com.y
        dz = resolved_system.z - old_com.z
        dvx = resolved_system.vx - old_com_velocity.x
        dvy = resolved_system.vy - old_com_velocity.y
        dvz = resolved_system.vz - old_com_velocity.z

        resolved_system.x = target.x + dx
        resolved_system.y = target.y + dy
        resolved_system.z = target.z + dz
        resolved_system.vx = target.vx + dvx
        resolved_system.vy = target.vy + dvy
        resolved_system.vz = target.vz + dvz

    return updated
