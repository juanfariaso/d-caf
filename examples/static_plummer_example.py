"""Minimal custom D-CAF model with a static embedded Plummer background."""

import numpy as np

from amuse.lab import (
    new_kroupa_mass_distribution,
    new_plummer_model,
    nbody_system,
    units,
)

from dcaf.backgroundgas.plummer import PlummerSphere
from dcaf.dcaf import DcafSystem
from dcaf.factory import FieldBinaryPopulation
from dcaf.framework import StarFormationFramework
from dcaf.utilities.parameters import get_default_configuration


class ConstantSFR(StarFormationFramework):
    def form_stars(self, active_stars):
        # This is the extension point for position- or state-dependent formation.
        return self.extract_next_event()


class InstantaneousExpulsionPlummer(PlummerSphere):
    def evolve_model(self, tend):
        if tend < self.t_ge:
            super().evolve_model(tend)
            return

        # Keep the embedded Plummer potential until t_ge, then remove it.
        super().evolve_model(self.t_ge)
        self.mtot = 0 | self.mtot.unit
        self.model_time = tend


def make_stars(total_mass, cloud_mass, gas_radius, nstars=300):
    masses = new_kroupa_mass_distribution(
        nstars,
        0.1 | units.MSun,
        100 | units.MSun,
    )
    masses *= total_mass / masses.sum()

    converter = nbody_system.nbody_to_si(cloud_mass, gas_radius)
    stars = new_plummer_model(nstars, convert_nbody=converter)
    stars.mass = masses

    binaries = FieldBinaryPopulation(
        mult_frac="field",
        pdist="inner",
        qdist="field",
        edist="field",
        min_mass=0.08,
    )
    stars = binaries.apply(stars)["resolved_stars"]
    return stars, converter


def main():
    np.random.seed(42)

    stellar_mass = 300 | units.MSun
    sfe = 0.3
    embedded_time = 1 | units.Myr
    gas_radius = 1 | units.parsec
    end_time = 5 | units.Myr

    cloud_mass = stellar_mass / sfe
    stars, converter = make_stars(stellar_mass, cloud_mass, gas_radius)

    gas = InstantaneousExpulsionPlummer(
        mtot=cloud_mass,
        rscale=gas_radius,
        mdot=0 | units.MSun / units.Myr,
        t0=0 | units.Myr,
        t_ge=embedded_time,
        t_col=0 | units.Myr,
        t_exp=1 | units.Myr,
    )

    framework = ConstantSFR(
        target_stars=stars,
        star_formation_rate=stars.mass.sum() / embedded_time,
        nstart=2,
        background_gas=gas,
    )

    config = get_default_configuration()
    config["petar"].number_of_workers = 1
    config["petar"].redirection = "file"
    dt_level = 15
    config["petar"].dt_soft = 2**-dt_level | nbody_system.time

    system = DcafSystem(
        framework=framework,
        config=config,
        converter=converter,
        output_folder="tutorial_output",
        track_background_gas_energy=False,
    )
    system.dt_out = 0.1 | units.Myr
    system.initialize_system()
    system.evolve_model(end_time)


if __name__ == "__main__":
    main()
