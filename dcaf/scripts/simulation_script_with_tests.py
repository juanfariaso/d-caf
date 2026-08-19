#!/usr/bin/env python3
"""
This script is similar to the simulation script we have, but is intended to test
new features. For instance, if we run singles and then binaries, the script
makes sure we have the same stellar population added at the same time.
Also, in order to test stellar evolution, we have a --flat-imf so we can see
stellar evolution evolve quickly for testing. Among other run tests that I will 
be adding.

note that by default we use a flat imf here. 
"""
import argparse
import os
import numpy as np
import yaml

from amuse.lab import nbody_system, new_kroupa_mass_distribution, units, new_plummer_model
from amuse.datamodel import Particles
from amuse.units.quantities import zero
from amuse.units.constants import G
try:
    from amuse.community.seba.interface import SeBa
    SEBA_INSTALLED = True
except:
    SEBA_INSTALLED = False

from dcaf.dcaf import DcafSystem
from dcaf.framework import StarFormationFramework
from dcaf.utilities.parameters import get_default_configuration
from dcaf.utilities.particles import get_key_positions
from dcaf.backgroundgas.plummer import PlummerSphere, test_plummer_evolution
from dcaf.factory import FieldBinaryPopulation

SEEDS = [7, 82, 36, 91, 54, 12, 68, 47, 99, 23]

class MyFormationFramework(StarFormationFramework):
    def form_stars(self, active_stars):
        """ 
        This is just an example. It does exactly the same as the original
        StarFormationFramework class. Overwrite if need something more complex
        """
        next_stars = self.extract_next_event()
        return next_stars


def sample_masses(
    Mtot,
    mmin=0.08 | units.MSun,
    mmax=100 | units.MSun,
    flat_imf=False,
    n_forced_massive=0,
    forced_mass=100 | units.MSun,
):
    """
    Sample stellar masses with either a standard or linear-flat IMF.

    The forced massive stars are counted within the total stellar mass budget.
    """
    if forced_mass > mmax:
        raise ValueError("forced_mass can not exceed the IMF upper limit.")

    if n_forced_massive < 0:
        raise ValueError("n_forced_massive must be non-negative.")

    forced = np.full(n_forced_massive, forced_mass.value_in(units.MSun)) | units.MSun
    remaining_mass = Mtot - forced.sum()

    if remaining_mass <= 0 | units.MSun:
        raise ValueError(
            "Total stellar mass is too small for the requested forced massive stars."
        )

    if flat_imf:
        mean_mass = 0.5 * (mmin + mmax)
        chunk_size = max(1, int((remaining_mass / mean_mass)))
        sampled_mass = 0 | units.MSun
        sampled_chunks = []

        while sampled_mass < 0.98 * remaining_mass:
            add = np.random.uniform(
                mmin.value_in(units.MSun),
                mmax.value_in(units.MSun),
                chunk_size,
            ) | units.MSun
            sampled_chunks.append(add.value_in(units.MSun))
            sampled_mass += add.sum()
    else:
        N = max(2, int((remaining_mass / (0.5 | units.MSun))))
        sampled = new_kroupa_mass_distribution(N, mmin, mmax)
        sampled_mass = sampled.sum()
        sampled_chunks = [sampled.value_in(units.MSun)]

        while sampled_mass < 0.98 * remaining_mass:
            add = new_kroupa_mass_distribution(N // 2 + 1, mmin, mmax)
            sampled_chunks.append(add.value_in(units.MSun))
            sampled_mass += add.sum()

    if n_forced_massive == 0:
        return np.concatenate(sampled_chunks) | units.MSun

    return np.concatenate([forced.value_in(units.MSun), *sampled_chunks]) | units.MSun

def get_tcol(Fmax, Mdot, M0, tau_emb, f=0.1):
    """ 
    Collapse timescale. We calculate the necessary timescale so the cloud
    reach a Fmax within the embedded timescale tau_emb. 
    """
    lambda_f = 1.0 / (f**(-2/3) - 1)**0.5
    mu = 1 + (Mdot / M0) * tau_emb
    term = (mu * lambda_f**3) / (f * Fmax)
    return tau_emb / (1 + lambda_f**2 - term**(2/3))


def apply_sigma_profile(stars, Mpl_gas0, Rpl_star, kappa, eta_sigma, Rpl_gas0):
    """
    Rescale stellar velocities so that the 1D dispersion follows:
        sigma(r) = sigma0 * (r/Rpl_star)^((2-kappa)/2)
    where:
        sigma0 = eta_sigma * sqrt(G*Mpl_gas0/Rpl_gas0)
    and sigma_pl is the equilibrium Plummer dispersion for (Mpl_gas0, Rpl_star).
    """
    r = (stars.x**2 + stars.y**2 + stars.z**2).sqrt()
    sigma_pl = ((G * Mpl_gas0) / (6.0 * Rpl_star) * (1.0 + (r / Rpl_star)**2)**(-0.5)).sqrt()
    sigma_vir0 = (G * Mpl_gas0 / Rpl_gas0).sqrt()
    sigma0 = float(eta_sigma) * sigma_vir0

    expo = 0.5 * (2.0 - float(kappa))
    eps = 1e-10 | Rpl_star.unit
    sigma_tgt = sigma0 * ((r + eps) / Rpl_star)**expo

    s = sigma_tgt / sigma_pl
    stars.vx *= s
    stars.vy *= s
    stars.vz *= s
    return stars


def build_stellar_distribution(
    masses,
    converter,
    Rpl_star,
    Rpl_gas0,
    Mpl_gas0,
    kappa,
    eta_sigma,
):
    """
    Create a fresh phase-space realization for the provided stellar masses.

    The input mass order is preserved exactly.
    """
    stars = new_plummer_model(len(masses), convert_nbody=converter)

    s_pos = Rpl_star / Rpl_gas0
    stars.x *= s_pos
    stars.y *= s_pos
    stars.z *= s_pos

    stars.mass = masses

    return apply_sigma_profile(
        stars,
        Mpl_gas0=Mpl_gas0,
        Rpl_star=Rpl_star,
        kappa=kappa,
        eta_sigma=eta_sigma,
        Rpl_gas0=Rpl_gas0,
    )


def run_seba_only(
    framework,
    config,
    t_end,
    output_folder="dcaf_output",
    min_timestep=None,
    adaptive_timestep=False,
):
    """
    Run only SeBa on the scheduled star-formation history.

    The base integration step is limited by framework.dt_tolerance, which is
    the minimum formation-time spacing used by the scheduler.

    If adaptive_timestep is True, SeBa can insert extra events using its own
    minimum particle timestep whenever that is shorter than the base step.
    """
    if not SEBA_INSTALLED:
        raise ImportError("SeBa not installed. We can not start seba-only runs")

    stars = framework.target_stars

    if not hasattr(stars, "birth_time"):
        stars.birth_time = -np.ones(len(stars)) | units.Myr
    if not hasattr(stars, "initial_mass"):
        stars.initial_mass = stars.mass.copy()
    if not hasattr(stars, "stellar_type"):
        stars.stellar_type = np.zeros(len(stars)) | units.stellar_type
    if not hasattr(stars, "relative_age"):
        stars.relative_age = np.zeros(len(stars)) | units.Myr
    if not hasattr(stars, "relative_mass"):
        stars.relative_mass = np.zeros(len(stars)) | units.MSun
    if not hasattr(stars, "core_mass"):
        stars.core_mass = np.zeros(len(stars)) | units.MSun
    if not hasattr(stars, "COcore_mass"):
        stars.COcore_mass = np.zeros(len(stars)) | units.MSun
    if not hasattr(stars, "in_seba"):
        stars.in_seba = np.zeros(len(stars), dtype=bool)

    seba = SeBa()
    seba.parameters.metallicity = config["stellar_evolution"].metallicity

    os.makedirs(output_folder, exist_ok=True)
    history_name = "seba_history_adaptive.csv" if adaptive_timestep else "seba_history_fixed.csv"
    history_path = os.path.join(output_folder, history_name)

    def sync_seba_to_target():
        if len(seba.particles) == 0:
            return

        seba_stars = seba.particles
        positions = get_key_positions(stars, seba_stars.key)
        target_seba_stars = stars[positions]
        target_seba_stars.mass = seba_stars.mass
        target_seba_stars.radius = seba_stars.radius
        target_seba_stars.stellar_type = seba_stars.stellar_type
        target_seba_stars.relative_age = seba_stars.relative_age
        target_seba_stars.relative_mass = seba_stars.relative_mass
        target_seba_stars.core_mass = seba_stars.core_mass
        target_seba_stars.COcore_mass = seba_stars.COcore_mass

    def register_new_stars(new_stars, model_time):
        if len(new_stars) == 0:
            return

        positions = get_key_positions(stars, new_stars.key)
        target_new_stars = stars[positions]
        target_new_stars.birth_time = model_time
        target_new_stars.initial_mass = target_new_stars.mass

        if np.any(target_new_stars.in_seba):
            raise ValueError(
                "[SEBA-ONLY] Attempted to register stars already present in SeBa."
            )

        target_new_stars.in_seba = True
        seba.particles.add_particles(target_new_stars.copy())

    def current_seba_timestep():
        if len(seba.particles) == 0:
            return None
        return min(seba.particles.time_step)

    def write_history_row(history_file, row_index, model_time):
        active_stars = stars[stars.is_active]
        if len(active_stars) == 0:
            total_mass = 0.0
            total_radius = 0.0
        else:
            total_mass = active_stars.mass.sum().value_in(units.MSun)
            total_radius = active_stars.radius.sum().value_in(units.RSun)

        history_file.write(
            f"{row_index},"
            f"{model_time.value_in(units.Myr):.16e},"
            f"{len(active_stars)},"
            f"{total_mass:.16e},"
            f"{total_radius:.16e}\n"
        )

    model_time = framework.get_next_formation_time()
    if model_time is None:
        raise ValueError("No formation events in framework")

    new_stars = framework.form_stars(Particles())
    register_new_stars(new_stars, model_time)
    sync_seba_to_target()

    dt_step = framework.dt_tolerance if min_timestep is None else min_timestep
    with open(history_path, "w") as history_file:
        history_file.write("step,time_myr,nstars,total_mass_msun,total_radius_rsun\n")

        row_index = 0
        write_history_row(history_file, row_index, model_time)
        row_index += 1

        while model_time < t_end:
            next_formation_time = framework.get_next_formation_time()
            next_step_time = model_time + dt_step

            candidate_times = [next_step_time, t_end]
            if next_formation_time is not None:
                candidate_times.append(next_formation_time)

            if adaptive_timestep:
                seba_dt = current_seba_timestep()
                if seba_dt is not None:
                    candidate_times.append(model_time + seba_dt)

            t_stop = min(candidate_times)

            if len(seba.particles) > 0 and t_stop > model_time:
                seba.evolve_model(t_stop)

            model_time = t_stop
            sync_seba_to_target()

            if next_formation_time is not None and model_time == next_formation_time:
                new_stars = framework.form_stars(stars[stars.is_active])
                register_new_stars(new_stars, model_time)
                sync_seba_to_target()

            write_history_row(history_file, row_index, model_time)
            row_index += 1

    seba.stop()


def main(
    seed_index=0,
    tff=None,
    Rcl=10.0 | units.pc,
    Rpl=7.0 | units.pc,
    Mstars=2000.0 | units.MSun,
    sfe=0.3,
    Fmax=None,
    tge_over_tff=1.0,
    texp_over_tff=1.0,
    t_ge=None,
    t_exp=None,
    mdot_factor=0.0,
    eta_radius=0.5,
    eta_sigma=0.6,
    kappa=1.8,
    nworkers=2,
    t_end=30.0 | units.Myr,
    dt_out=0.05 | units.Myr,
    dt_level=15,
    stars_per_worker=0,
    track_background_gas_energy=False,
    test_background=False,
    dry_run=False,
    field_binaries=False,
    flat_imf=False,
    n_forced_massive=0,
    forced_mass=100 | units.MSun,
    seba_only=False,
    seba_min_timestep=None,
    seba_adaptive_timestep=False,
    resume=False,
    stellar_evolution=False,
    metallicity=0.02,
):

    if dry_run:
        print("exit OK")
        return

    if stars_per_worker == 0:
        stars_per_worker = None

    np.random.seed(SEEDS[seed_index])

    Mcl = Mstars / sfe

    # Keep the original shape of the cloud family fixed.
    f_pl = Rpl / Rcl

    # Optional tff override:
    # if given, derive Rcl from tff and rescale the whole system homologously.
    if tff is not None:
        Rcl = ((8.0 * G * Mcl * tff**2) / (np.pi**2))**(1.0 / 3.0)
        Rpl = f_pl * Rcl
    else:
        rho_mean = 3.0 * Mcl / (4.0 * np.pi * Rcl**3)
        tff = ((3.0 * np.pi) / (32.0 * G * rho_mean))**0.5

    if t_ge is None:
        t_ge = float(tge_over_tff) * tff
    if t_exp is None:
        t_exp = float(texp_over_tff) * tff

    mass_fraction = Rcl**3 / (Rcl**2 + Rpl**2)**1.5
    Mpl = Mcl / mass_fraction

    star_formation_rate = Mstars / t_ge
    mdot = star_formation_rate * mdot_factor

    t_col = zero
    if Fmax is not None:
        t_col = get_tcol(Fmax, mdot, Mpl, t_ge)

    masses = sample_masses(
        Mstars,
        flat_imf=flat_imf,
        n_forced_massive=n_forced_massive,
        forced_mass=forced_mass,
    )
    ntot = len(masses)

    # sample stars in gas dynamical units
    Rpl_gas0 = Rpl
    Mpl_gas0 = Mpl
    Rpl_star = float(eta_radius) * Rpl_gas0

    converter = nbody_system.nbody_to_si(Mpl_gas0, Rpl_gas0)
    target_stars = build_stellar_distribution(
        masses,
        converter=converter,
        Rpl_star=Rpl_star,
        Rpl_gas0=Rpl_gas0,
        Mpl_gas0=Mpl_gas0,
        kappa=kappa,
        eta_sigma=eta_sigma,
    )

    # In this comparison script we always build the same resolved mass list.
    # If `field_binaries` is False, the companion masses are still created, but
    # they are resampled as independent stars and their `system_id` is reset so
    # they are not scheduled or evolved as linked systems.
    binary_population = FieldBinaryPopulation(
        mult_frac='field',
        pdist='inner',
        qdist='field',
        edist='field',
        min_mass=0.08
    )
    binary_data = binary_population.apply(target_stars)

    if field_binaries:
        target_stars = binary_data['resolved_stars']
    else:
        comparison_masses = binary_data['resolved_stars'].mass.copy()
        target_stars = build_stellar_distribution(
            comparison_masses,
            converter=converter,
            Rpl_star=Rpl_star,
            Rpl_gas0=Rpl_gas0,
            Mpl_gas0=Mpl_gas0,
            kappa=kappa,
            eta_sigma=eta_sigma,
        )
        target_stars.system_id = -np.ones(len(target_stars), dtype=int)
        if hasattr(target_stars, "secondary_id"):
            target_stars.secondary_id = -np.ones(len(target_stars), dtype=int)
        if hasattr(target_stars, "is_secondary"):
            target_stars.is_secondary = np.zeros(len(target_stars), dtype=bool)
        if hasattr(target_stars, "binary_period"):
            target_stars.binary_period = np.zeros(len(target_stars)) | units.day


    cloud = PlummerSphere(
        mtot=Mpl,
        rscale=Rpl,
        t0=zero,
        t_ge=t_ge,
        t_col=t_col,
        t_exp=t_exp,
        mdot=mdot
    )

    if test_background:
        test_plummer_evolution(
            model=cloud,
            times=np.linspace(0, t_end.value_in(units.Myr), 50) | units.Myr
        )
        return

    print(f"Mcl = {Mcl.in_(units.MSun)}")
    print(f"Rcl = {Rcl.in_(units.pc)}")
    print(f"Rpl = {Rpl.in_(units.pc)}")
    print(f"tff = {tff.in_(units.Myr)}")
    print(f"t_ge = {t_ge.in_(units.Myr)}")
    print(f"t_exp = {t_exp.in_(units.Myr)}")

    framework = MyFormationFramework(
        target_stars,
        star_formation_rate=star_formation_rate,
        nstart=10,
        background_gas=cloud
    )

    framework.dt_tolerance = converter.to_si(2**-10 | nbody_system.time)
    framework.schedule_formation()

    cfg = get_default_configuration()
    cfg["petar"].number_of_workers = nworkers
    cfg["petar"].redirection = "file"
    cfg["petar"].r_out = 0.03 | units.pc
    cfg["petar"].r_bin = 1000 | units.au
    cfg["petar"].dt_soft = 2**(-dt_level) | nbody_system.time
    cfg["stellar_evolution"].metallicity = metallicity

    if seba_only:
        run_seba_only(
            framework,
            config=cfg,
            t_end=t_end,
            output_folder="dcaf_output",
            min_timestep=seba_min_timestep,
            adaptive_timestep=seba_adaptive_timestep,
        )
    else:
        system = DcafSystem(
            framework,
            converter=converter,
            config=cfg,
            gas_code=cloud,
            stellar_evolution=stellar_evolution,
            track_background_gas_energy=track_background_gas_energy,
            stars_per_worker=stars_per_worker,
            resume=resume
        )

        dt_out_nbody = 2**round(np.log2(dt_out / converter.to_si(1.0 | nbody_system.time))) | nbody_system.time
        system.dt_out = converter.to_si(dt_out_nbody)

        system.initialize_system()
        system.evolve_model(t_end)


# ---------------- YAML + CLI glue ----------------

def to_quantity(x, unit):
    return float(x) | unit


PARAMETER_SPECS = {
    "seed_index": {"cast": int},
    "tff": {"unit": units.Myr},
    "Rcl": {"unit": units.pc},
    "Rpl": {"unit": units.pc},
    "Mstars": {"unit": units.MSun},
    "sfe": {"cast": float},
    "Fmax": {"cast": float, "allow_none": True},
    "tge_over_tff": {"cast": float},
    "texp_over_tff": {"cast": float},
    "t_ge": {"unit": units.Myr},
    "t_exp": {"unit": units.Myr},
    "mdot_factor": {"cast": float},
    "eta_radius": {"cast": float},
    "eta_sigma": {"cast": float},
    "kappa": {"cast": float},
    "nworkers": {"cast": int},
    "t_end": {"unit": units.Myr},
    "dt_out": {"unit": units.Myr},
    "dt_level": {"cast": int},
    "stars_per_worker": {"cast": int},
    "track_background_gas_energy": {"cast": bool},
    "test_background": {"cast": bool},
    "field_binaries": {"cast": bool},
    "flat_imf": {"cast": bool},
    "n_forced_massive": {"cast": int, "allow_none": True},
    "forced_mass": {"unit": units.MSun},
    "seba_only": {"cast": bool},
    "seba_min_timestep": {"unit": units.Myr, "allow_none": True},
    "seba_adaptive_timestep": {"cast": bool},
    "dry_run": {"cast": bool},
    "stellar_evolution": {"cast": bool},
    "metallicity": {"cast": float},
}


def get_default_params():
    return dict(
        seed_index=0,
        tff=None,
        Rcl=10.0 | units.pc,
        Rpl=7.0 | units.pc,
        Mstars=2000.0 | units.MSun,
        sfe=0.3,
        Fmax=None,
        tge_over_tff=1.0,
        texp_over_tff=1.0,
        t_ge=None,
        t_exp=None,
        mdot_factor=0.0,
        eta_radius=0.5,
        eta_sigma=0.6,
        kappa=1.8,
        nworkers=2,
        t_end=30.0 | units.Myr,
        dt_out=0.05 | units.Myr,
        dt_level=15,
        stars_per_worker=0,
        track_background_gas_energy=False,
        test_background=False,
        dry_run=False,
        field_binaries=False,
        flat_imf=False,
        n_forced_massive=0,
        forced_mass=100 | units.MSun,
        seba_only=False,
        seba_min_timestep=None,
        seba_adaptive_timestep=False,
        resume=False,
        stellar_evolution=False,
        metallicity=0.02,
    )


def apply_overrides(params, source, *, from_cli=False, keys=None):
    if keys is None:
        keys = PARAMETER_SPECS.keys()

    for key in keys:
        if from_cli:
            raw_value = getattr(source, key)
            if raw_value is None:
                continue
        else:
            if key not in source:
                continue
            raw_value = source[key]
            if raw_value is None and not PARAMETER_SPECS[key].get("allow_none"):
                continue

        spec = PARAMETER_SPECS[key]
        if spec.get("allow_none") and raw_value is None:
            params[key] = None
        elif "unit" in spec:
            value = float(raw_value) if from_cli else float(str(raw_value).split()[0])
            params[key] = to_quantity(value, spec["unit"])
        else:
            params[key] = spec["cast"](raw_value)


def load_yaml_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def build_parser():
    p = argparse.ArgumentParser(description="DCAF parametric Plummer background run (YAML-capable).")
    p.add_argument("--config", type=str, default=None, help="YAML config file")
    p.add_argument("--resume", action="store_true", help="Resume from current folder using config.yaml (or --config)")

    p.add_argument("--seed_index", type=int, default=None)
    p.add_argument("--tff", type=float, default=None, help="optional target cloud free-fall time [Myr]")
    p.add_argument("--tge_over_tff", type=float, default=None)
    p.add_argument("--texp_over_tff", type=float, default=None)

    p.add_argument("--t_ge", type=float, default=None, help="override t_ge [Myr]")
    p.add_argument("--t_exp", type=float, default=None, help="override t_exp [Myr]")

    p.add_argument("--Rcl", type=float, default=None, help="cloud radius [pc]")
    p.add_argument("--Rpl", type=float, default=None, help="gas Plummer scale radius [pc]")

    p.add_argument("--Mstars", type=float, default=None, help="target stellar mass [Msun]")
    p.add_argument("--sfe", type=float, default=None)
    p.add_argument("--Fmax", type=float, default=None)

    p.add_argument("--eta_radius", type=float, default=None)
    p.add_argument("--eta_sigma", type=float, default=None)
    p.add_argument("--kappa", type=float, default=None)

    p.add_argument("--mdot_factor", type=float, default=None)
    p.add_argument("--nworkers", type=int, default=None)
    p.add_argument("--t_end", type=float, default=None)
    p.add_argument("--dt_out", type=float, default=None)
    p.add_argument("--dt_level", type=int, default=None)
    p.add_argument("--stars_per_worker", type=int, default=None)

    p.add_argument("--test_background", action="store_true")
    p.add_argument("--track_background_gas_energy", action="store_true")
    p.add_argument("--field_binaries", action="store_true")
    p.add_argument("--flat-imf", action="store_true")
    p.add_argument("--n-forced-massive", type=int, default=None)
    p.add_argument("--forced-mass", type=float, default=None, help="forced stellar mass [Msun]")
    p.add_argument("--seba-only", action="store_true")
    p.add_argument("--seba-min-timestep", type=float, default=None, help="minimum SeBa-only step [Myr]")
    p.add_argument("--seba-adaptive-timestep", action="store_true")
    p.add_argument("--stellar_evolution", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--metallicity", type=float, default=None)

    return p


def parse_args():
    a = build_parser().parse_args()

    params = get_default_params()

    resume_cli_keys = ("t_end",)
    normal_cli_keys = (
        "seed_index",
        "tff",
        "Rcl",
        "Rpl",
        "Mstars",
        "sfe",
        "Fmax",
        "tge_over_tff",
        "texp_over_tff",
        "t_ge",
        "t_exp",
        "mdot_factor",
        "eta_radius",
        "eta_sigma",
        "kappa",
        "nworkers",
        "t_end",
        "dt_out",
        "dt_level",
        "stars_per_worker",
        "metallicity",
        "n_forced_massive",
        "forced_mass",
        "seba_min_timestep",
    )

    if a.resume:
        cfg_path = a.config if a.config is not None else "config.yaml"
        apply_overrides(params, load_yaml_config(cfg_path))

        params["resume"] = True
        apply_overrides(params, a, from_cli=True, keys=resume_cli_keys)

        if a.test_background:
            params["test_background"] = True
        if a.track_background_gas_energy:
            params["track_background_gas_energy"] = True
        if a.flat_imf:
            params["flat_imf"] = True
        if a.seba_only:
            params["seba_only"] = True
        if a.seba_adaptive_timestep:
            params["seba_adaptive_timestep"] = True
        if a.dry_run:
            params["dry_run"] = True

        return params

    y = {}
    if a.config is not None:
        y = load_yaml_config(a.config)

    if y:
        apply_overrides(params, y)

    apply_overrides(params, a, from_cli=True, keys=normal_cli_keys)

    if a.test_background: params["test_background"] = True
    if a.track_background_gas_energy: params["track_background_gas_energy"] = True
    if a.field_binaries: params["field_binaries"] = True
    if a.flat_imf: params["flat_imf"] = True
    if a.seba_only: params["seba_only"] = True
    if a.seba_adaptive_timestep: params["seba_adaptive_timestep"] = True
    if a.stellar_evolution: params["stellar_evolution"] = True
    if a.dry_run: params["dry_run"] = True

    return params


def save_params(params, filename="config.yaml"):
    def serialise(val):
        if hasattr(val, "unit"):
            return f"{val.value_in(val.unit)} {val.unit}"
        return val

    data = {k: serialise(v) for k, v in params.items()}
    with open(filename, "w") as f:
        yaml.dump(data, f, sort_keys=False)
    print(f"Saved parameters to {filename}")


if __name__ == "__main__":
    params = parse_args()

    resume = params.get("resume", False)
    if not resume:
        save_params(params)

    main(**params)
