"""

This script is an example on how to generate binaries and triples using the
class defined in d-caf.
The basic steps are:

1. define a BinaryPopulation subclass. Here you will redefine the functions to
select companions and binary properties.
2. build the initial set of stars (We will pair these systems to make B and T.
3. apply the class once to resolve binaries (create the binaries)
4. apply it again on the unresolved systems to turn some binaries into triples.
"""

import numpy as np
from amuse.lab import new_kroupa_mass_distribution, new_plummer_model, nbody_system, units
from dcaf.factory import BinaryPopulation


class CustomTriplePopulation(BinaryPopulation):
    """
    Example custom class.

    - binary companions are chosen randomly from the remaining mass pool,
    - periods are sampled at random in log-space,
    - eccentricities are sampled uniformly,
    - orbital-plane orientations are left to the internal random sampling,
    - when re-applying on unresolved systems, a random single is attached to a
      random binary to make a triple.


    Below the functions that need to be rewritten if want to be customized.
    Otherwise it uses the default at BinaryPopulation
    """

    def choose_companion(self, m1, mass, equal_mass=False):
        """
        this function is meant to select a companion for m1 from the mass list.
        Here just picking a random.
        """
        return mass[np.random.randint(len(mass))]

    def select_binaries(self, stars, nbinaries):
        """
        This function pairs the population using the choose_compation function.

        It must retrn the indexes list of primaries and secondaries
        """
        mass = stars.mass
        available_indexes = list(range(len(stars)))
        primary_index = []
        companion_index = []

        for _ in range(nbinaries):
            if len(available_indexes) < 2:
                break

            primary = int(np.random.choice(available_indexes))
            available_indexes.remove(primary)

            companion_mass = self.choose_companion(
                mass[primary],
                mass[available_indexes],
                equal_mass=False,
            )

            candidate_indexes = [
                index for index in available_indexes if mass[index] == companion_mass
            ]
            companion = int(np.random.choice(candidate_indexes))
            available_indexes.remove(companion)

            if mass[companion] > mass[primary]:
                primary, companion = companion, primary

            primary_index.append(primary)
            companion_index.append(companion)

        return primary_index, companion_index

    def get_periods(self, stars, primary_index, companion_index, **kwargs):
        """
        Select the binaries period distribution. Here just random
        """
        logpmax = kwargs.get("logpmax", 5.0)
        logpmin = kwargs.get("logpmin", 1.0)
        log10_period = np.random.uniform(logpmin, logpmax, size=len(primary_index))
        return 10.0 ** log10_period | units.day

    def get_eccentricities(self, stars, primary_index, companion_index, **kwargs):
        """
        Radom eccentricities as example. This is also called on the first pass
        """
        # Random binary eccentricities between 0 and 0.9.
        return np.random.uniform(0.0, 0.9, size=len(primary_index))

    def select_higher_order_companion(
        self, primary_index, stars, hierarchy, available_indexes, **kwargs
    ):
        """
        This function is called if we do a second pass on the population.
        It select the companion for the system as in choose_companion, but for
        higher orders.
        We just return attach a single to an existing binary.
        Otherwise return None
        """
        if len(hierarchy[primary_index]["members"]) != 2:
            return None

        single_indexes = [
            index
            for index in available_indexes
            if index != primary_index and len(hierarchy[index]["members"]) == 1
        ]
        if len(single_indexes) == 0:
            return None

        return int(np.random.choice(single_indexes))

    def get_higher_order_periods(
        self, stars, primary_index, companion_index, hierarchy, **kwargs
    ):
        """
        Sample random outer periods for triples.

        Here we just a random order 10 higher period distribution

        """
        periods = self.get_periods(
                stars,
                primary_index,
                companion_index,
                logpmin=5.0,
                logpmax=8.0,
            )

        return periods

    def get_higher_order_eccentricities(
        self, stars, primary_index, companion_index, hierarchy, **kwargs
    ):
        """
        Same as above, but for eccentricities
        """
        return np.random.uniform(0.0, 0.9, size=len(primary_index))


def summarize(result, label):
    counts = {}
    for entry in result["hierarchy"]:
        order = len(entry["members"])
        counts[order] = counts.get(order, 0) + 1

    print(label)
    print(f"  resolved stars: {len(result['resolved_stars'])}")
    print(f"  unresolved systems: {len(result['unresolved_stars'])}")
    print(f"  singles: {counts.get(1, 0)}")
    print(f"  binaries: {counts.get(2, 0)}")
    print(f"  triples: {counts.get(3, 0)}")


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

# 1) Generate an initial set of single stars with positions, velocities, and masses.
nstars = 128
radius = 0.5 | units.pc
masses = new_kroupa_mass_distribution(
    nstars,
    mass_min=0.08 | units.MSun,
    mass_max=20.0 | units.MSun,
)
converter = nbody_system.nbody_to_si(masses.sum(), radius)

stars = new_plummer_model(nstars, convert_nbody=converter)
stars.mass = masses
stars.radius = 0.0 | units.RSun

# 2) Instantiate the user-defined population model.
population = CustomTriplePopulation(
    nbinaries=24,
    max_semi_major_axis=radius,
)

# 3) First pass: resolve binaries from the original star catalog.
binaries = population.apply(stars)
resolved_binaries = binaries["resolved_stars"]
unresolved_binaries = binaries["unresolved_stars"]
hierarchy_after_binaries = binaries["hierarchy"]

summarize(binaries, "After first pass (binaries)")

# 4) Second pass: apply the same class again, but now on the unresolved systems.
#    The `hierarchy` argument tells the code how to expand the old binary members
#    back into resolved stars after adding an outer companion.
triples = population.apply(
    unresolved_binaries,
    hierarchy=hierarchy_after_binaries,
    force_n_binaries=8,
)

resolved_triples = triples["resolved_stars"]
unresolved_triples = triples["unresolved_stars"]
hierarchy_after_triples = triples["hierarchy"]

summarize(triples, "After second pass (triples)")

# At this point:
# - `resolved_binaries` / `resolved_triples` contain the particle-level ICs,
# - `unresolved_binaries` / `unresolved_triples` contain one particle per system,
# - `hierarchy_after_binaries` / `hierarchy_after_triples` describe membership
#   and orbits for each unresolved system.
