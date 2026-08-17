"""
D-CAF: Dynamic Cluster Assembly Framework
"""
from __future__ import annotations
import os
import shutil
import numpy as np

from amuse.datamodel import Particles
from amuse.units.constants import G
from amuse.units import units, nbody_system
from amuse.couple.bridge import Bridge
try: 
    from amuse.community.petar.interface import Petar
    PETAR_INSTALLED = True
except:
    print('Warning PeTar not installed, can not run simulations' )
    PETAR_INSTALLED = False

try: 
    from amuse.community.seba.interface import SeBa
    SEBA_INSTALLED = True
except:
    print('Warning SEBA not installed, can not initalize stellar evolution' )
    SEBA_INSTALLED = False

from amuse.io import write_set_to_file

from dcaf.utilities.parameters import get_default_configuration
from dcaf.utilities.logger import setup_logger
from dcaf.utilities.particles import get_key_positions

from dcaf.io.restart import get_resume_state
from dcaf.io.output import get_output_folders, find_snapshot_files

# from dcaf.framework import StarFormationFramework  

class DcafSystem:
    def __init__(
        self,
        framework,# requires: target_stars, next_formation_time(), form_stars()
        config = None,
        converter = None,
        gas_code = None,
        track_background_gas_energy = True, 
        output_folder = "./dcaf_output/",
        log_level = 'debug',
        stars_per_worker = None,
        workers_step = 5,
        stellar_evolution = False,
        resume = False
    ):
        base_output_folder = output_folder.rstrip("/")
        if base_output_folder == "":
            base_output_folder = "."

        self.snapshot_basename = "stars_"
        reuse_empty_output_folder = False

        self.stellar_evolution = stellar_evolution
        if self.stellar_evolution and not SEBA_INSTALLED:
            raise ImportError("SeBa not installed. We can not start stellar evolution")

        self.resume = resume
        if self.resume:
            folders = get_output_folders(base_output_folder)
            latest_folder = folders[-1]
            latest_snapshots = find_snapshot_files(
                source_folder=latest_folder,
                snapshot_basename=self.snapshot_basename,
            )

            if len(latest_snapshots) > 0:
                seg = len(folders)
                self.resume_source_folder = latest_folder
                self.output_folder = f"{base_output_folder}_{seg}"
            else:
                if len(folders) < 2:
                    raise FileNotFoundError(
                        f"No previous data to resume in '{base_output_folder}'."
                    )
                previous_folder = folders[-2]
                previous_snapshots = find_snapshot_files(
                    source_folder=previous_folder,
                    snapshot_basename=self.snapshot_basename,
                )
                if len(previous_snapshots) == 0:
                    raise FileNotFoundError(
                        "Found two consecutive empty output folders: "
                        f"'{previous_folder}' and '{latest_folder}'."
                        " Please check, and remove folders with no snapshots."
                    )
                seg = len(folders) - 1
                self.resume_source_folder = previous_folder
                self.output_folder = latest_folder
                reuse_empty_output_folder = True
        else:
            seg = 0
            self.output_folder = base_output_folder

        #where to get the old data
        if not self.resume:
            self.resume_source_folder = None

        self.config = config or get_default_configuration()
        self.workers_step = workers_step # how many workers to increase

        self.framework = framework

        self.dt_out = 0.5 | units.Myr   # TODO: move to config
        self.__current_snapshot = 0
        self.stars_per_worker = stars_per_worker
        self._current_workers = None

        # I will move this inside dcaf_output in the future..
        self.gas_code = gas_code
        if self.framework.background_gas:
            self.gas_code = self.framework.background_gas
            if seg == 0:
                self.gas_code.logfilename = "background_gas.dat"
            else:
                self.gas_code.logfilename = f"background_gas_{seg}.dat"

        if reuse_empty_output_folder:
            log_file = os.path.join(self.output_folder, "dcaf.log")
            failed_log_file = os.path.join(self.output_folder, "dcaf.log.failed")
            if os.path.exists(log_file):
                shutil.copyfile(log_file, failed_log_file)

        self.logger = setup_logger(self.output_folder,log_level)
        # Converter: use provided one or derive from framework target stars
        stars0 = framework.target_stars
        if len(stars0) < 2:
            raise ValueError('Framework target stars must have at least two \
            particles')
        if converter is None:
            converter = nbody_system.nbody_to_si(stars0.total_mass(),\
                    stars0.virial_radius())
        self.converter = converter

        # Runtime state
        self.target_stars = stars0
        self.model_time = None

        # Codes 
        self.petar_code = None
        self.seba_code = None
        self.bridge_code = None
        self.code = None

        # Particle channel
        self._channel_code_to_mem = None

        self.dt_soft_eff = None

        # Energy checks
        self._total_energy = [0 |units.J,0 | units.J]  # [previous, current]
        # Energy components for the stars
        # [Tstars, Ustars_self, U_stars_gas]
        self._energy_components = [0 | units.J]*3
        # Cumulative individual energy budgets
        # [ E_new_stars, E_SE, E_gas_evol, E_binaries ]
        self._energy_budgets = [0 | units.J]*4
        self._energy_header_written = False
        # last step, cumulative
        self._energy_errors = [0, 0]
        self._gas_tracker = None
        self._last_energy_check = None
        self.track_background_gas_energy = track_background_gas_energy

    @property
    def formed_stars(self):
        """
        Return the active stars.

        self.target_stars contain all stars that will be in the simulation.
        This function just make it return the ones that have already formed and
        are updated by the evolving codes.
        Before giving th stars, we synchornize the active stars with the ones in
        the codes.
        """
        active_mask = self.target_stars.is_active

        if self.petar_code is not None:
            self.update_formed_stars()

        return self.target_stars[active_mask]

    def update_formed_stars(self):
        with self.logger.timing('[UPDATING STARS] ********************'):
            active_mask = self.target_stars.is_active
            active_stars = self.target_stars[active_mask]
            code_stars = self.petar_code.particles

            assert len(active_stars) == len(code_stars), (
                "[DCAF][SYNC] Active target stars and PeTar particles have "
                f"different lengths: {len(active_stars)} != {len(code_stars)}"
            )

            indices = get_key_positions(code_stars, active_stars.key)
            code_match = code_stars[indices]

            active_stars.x = code_match.x
            active_stars.y = code_match.y
            active_stars.z = code_match.z
            active_stars.vx = code_match.vx
            active_stars.vy = code_match.vy
            active_stars.vz = code_match.vz

            self.target_stars.collection_attributes.code_time = self.petar_code.model_time

    def initialize_system(self):
        """Instantiate PeTar and, if present, Bridge. Also add initial stars,
        or resume from the latest saved snapshot if self.resume is True.
        """
        if not PETAR_INSTALLED:
            print('PeTar not installed. System not initialized')
            return

        with self.logger.timing('[DCAF] Initializing system *********************'):
            # validate dt_soft
            dt_soft = self.config["petar"].dt_soft
            if dt_soft is None:
                raise ValueError('dt_soft must be provided in this implemenation')

            if not (
                dt_soft.unit == nbody_system.time and
                np.isclose(
                    np.log2(dt_soft.value_in(nbody_system.time)),
                    round(np.log2(dt_soft.value_in(nbody_system.time))),
                    atol=1e-12
                )
            ):
                raise ValueError(
                    "[DCAF] dt_soft must be in nbody_system.time units and an "
                    "exact power of 2."
                )

            self.dt_soft_eff = self.converter.to_si(dt_soft)

            # fresh runs need a valid formation schedule
            if not self.resume:
                self._validate_formation_schedule(dt_soft)

            with self.logger.timing('Initializing PeTar'):
                self._setup_petar()

            if self.stellar_evolution:
                with self.logger.timing('Initializing SeBa'):
                    self._setup_seba()

                    stars = self.target_stars

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

            # Make sure output directory exists
            os.makedirs(self.output_folder, exist_ok=True)

            # ------------------------------------------------------------
            # Resume branch
            # ------------------------------------------------------------
            if self.resume:
                state = get_resume_state(
                    framework=self.framework,
                    source_folder=self.resume_source_folder,
                    snapshot_basename=self.snapshot_basename,
                )

                self.logger.info(
                    f"[DCAF] Resume source folder: {state['source_folder']}"
                )

                stars = state["stars"]
                self.model_time = state["model_time"]
                snapshot_index = state["snapshot_index"]

                self.logger.info(
                    f"[DCAF] Resuming from snapshot {snapshot_index:03d} "
                    f"at {self.model_time.in_(units.Myr)}"
                )

                assert len(self.target_stars) == len(stars), (
                    "[DCAF][RESUME] Target stars and resumed stars have "
                    f"different lengths: {len(self.target_stars)} != {len(stars)}"
                )

                # copy the framework target stars (original) to the current set.
                # note that both should point to the same set which is the 
                # authoritative copy.
                self.target_stars = self.target_stars.copy_to_new_particles(
                    keys=stars.key
                )
                self.framework.target_stars = self.target_stars

                # the resume stars are the old state stars
                # so now we update the properties of the framework target stars
                # with the advanced stored set.
                resume_match = stars

                self.target_stars.is_active = True
                self.target_stars.x = resume_match.x
                self.target_stars.y = resume_match.y
                self.target_stars.z = resume_match.z
                self.target_stars.vx = resume_match.vx
                self.target_stars.vy = resume_match.vy
                self.target_stars.vz = resume_match.vz
                self.target_stars.mass = resume_match.mass

                if hasattr(stars, "radius"):
                    self.target_stars.radius = resume_match.radius

                if self.stellar_evolution:
                    required_attrs = (
                        "birth_time",
                        "initial_mass",
                        "in_seba",
                        "stellar_type",
                        "relative_age",
                        "relative_mass",
                        "core_mass",
                        "COcore_mass",
                    )
                    missing = [
                        name for name in required_attrs if not hasattr(stars, name)
                    ]
                    if missing:
                        raise ValueError(
                            "Resume with stellar evolution requires snapshots "
                            "to contain SeBa restart fields. Missing: "
                            + ", ".join(missing)
                        )

                    self.target_stars.birth_time = resume_match.birth_time
                    self.target_stars.initial_mass = resume_match.initial_mass
                    self.target_stars.in_seba = resume_match.in_seba
                    self.target_stars.stellar_type = resume_match.stellar_type
                    self.target_stars.relative_age = resume_match.relative_age
                    self.target_stars.relative_mass = resume_match.relative_mass
                    self.target_stars.core_mass = resume_match.core_mass
                    self.target_stars.COcore_mass = resume_match.COcore_mass

                self.petar_code.particles.add_particles(self.target_stars.copy())

                if self.stellar_evolution:
                    self.seba_code.evolve_model(self.model_time)
                    self.seba_code.particles.add_particles(self.target_stars.copy())

                # disable formation. We assume star formation is over in the
                # resume case.
                # TODO: Make resume branch to work during formation.
                self.framework.formation_sequence = []
                self.framework.formation_times = []
                self.framework._StarFormationFramework__next_formation_time = None
                self.framework._StarFormationFramework__next_stars = None

                self.__current_snapshot = snapshot_index + 1

            # ------------------------------------------------------------
            # Fresh initialization branch
            # ------------------------------------------------------------
            else:
                # get first formation event
                tnext = self.framework.get_next_formation_time()
                if tnext is None:
                    raise Exception('No formation events in framework')

                tnext = self._ceil_to_block(tnext)
                newstars = self.framework.form_stars(Particles())

                self.model_time = tnext
                if self.stellar_evolution:
                    # get seba into the same time as the rest
                    # note that on petar we do this by setting its initial time
                    # to model time. Here we just advance an empty seba
                    self.seba_code.evolve_model(self.model_time)
                self._add_new_stars(newstars)

            # ------------------------------------------------------------
            # Common tail
            # ------------------------------------------------------------
            self.dt_soft_eff = self.petar_code.parameters.dt_soft
            self.logger.info(
                '[DCAF] effective dt_soft changed from'
                f' {self.config["petar"].dt_soft}'
                f' {self.converter.to_nbody(self.dt_soft_eff)}'
                f' ({self.dt_soft_eff.in_(units.Myr)})'
            )

            if abs(
                self.converter.to_nbody(self.dt_soft_eff)
                - self.config["petar"].dt_soft
            ) > 1e-15 | nbody_system.time:
                raise Exception(
                    '[DCAF] effective dt_soft changed from'
                    f' {self.config["petar"].dt_soft}'
                    f' {self.converter.to_nbody(self.dt_soft_eff)}'
                    ' This may happened if dt_soft was not provided in nbody'
                    ' units'
                )

            self.petar_code.parameters.begin_time = self.model_time

            # also advance background gas
            if self.gas_code:
                self.logger.info(
                    '[DCAF] [BGAS] evolved to '
                    f'{self.model_time.in_(units.Myr)}'
                )
                self.gas_code.evolve_model(self.model_time)

            # initialize bridge
            if self.gas_code is not None:
                n_timestep = 1
                self._setup_bridge()
                self.bridge_code.time = self.model_time

                self.logger.info(
                    '[DCAF] [BRIDGE] setup with effective time-step: '
                    f'{(self.dt_soft_eff * n_timestep).in_(units.Myr)}'
                )

                self.bridge_code.add_system(self.petar_code, (self.gas_code,), False)
                self.bridge_code.add_system(self.gas_code,)

                if self.track_background_gas_energy:
                    self._gas_tracker = GasEnergyTracker(self)
                    self.bridge_code.add_system(self._gas_tracker)

                self.code = self.bridge_code
            else:
                self.logger.info(
                    '[DCAF][BRIDGE] no BGAS found. Evolving only with PeTar'
                )
                self.bridge_code = None
                self.code = self.petar_code

        # only write initial output for fresh runs
        if not self.resume:
            self.write_output()

    # --- setup helpers -----------------------------------------------------
    def _setup_petar(self,nworkers = None):
        """Initialize PeTar. No particles are added here.
            nworkers: if not given, is retrieved from config
        """
        cfg = self.config["petar"]
        if nworkers is None:
            nworkers = cfg.number_of_workers
        self.petar_code = Petar(self.converter,mode = 'gpu',
                                redirection = cfg.redirection,
                                number_of_workers = nworkers )
        self._current_workers = nworkers

        self.petar_code.parameters.theta = cfg.theta
        self.petar_code.parameters.r_bin = cfg.r_bin
        self.petar_code.parameters.r_out = cfg.r_out 
        self.petar_code.parameters.dt_soft = cfg.dt_soft


    def _setup_seba(self):
        """
        Initialize SeBa. No particles are added here.
        """

        if "stellar_evolution" not in self.config:
            raise ValueError(
                "stellar_evolution=True requires a 'stellar_evolution' "
                "configuration block."
            )

        self.seba_code = SeBa()

        cfg = self.config["stellar_evolution"]
        self.seba_code.parameters.metallicity = cfg.metallicity

    def _setup_bridge(self):
        cfg = self.config["bridge"]
        if cfg.timestep is None:
            timestep = self.dt_soft_eff
        else:
            timestep = cfg.timestep


        self.bridge_code = Bridge(timestep = timestep,
                                  use_threading=cfg.use_threading,
                                  verbose=cfg.verbose)

    def setup_gas(self):
        # TODO: add setup gas routine to StarFormationFramework
        #   I think we dont need this with the current implementation
        pass


    # -- helpers for the restart --------------------------------------------

    def _rebuild_system(self, nworkers):
        """
        Stop current PeTar, create a new instance with nworkers, and re-attach
        it (and Bridge if present) with the existing particles and model_time.
        """
        # grab current state
        old_petar = self.petar_code
        stars     = old_petar.particles.copy()
        t_now     = old_petar.model_time

        # stop old instance
        old_petar.stop()
        if self.bridge_code :
            self.bridge_code.stop()

        # update config so future calls know the current worker count
        # self.config["petar"].number_of_workers = nworkers

        # build new PeTar with same converter/params but new worker count
        self.logger.info(f"[DCAF][SCALING] Rebuilding PeTar with {nworkers} workers")
        self._setup_petar(nworkers = nworkers)  # uses self.config["petar"]

        # restore particles and time
        self.petar_code.particles.add_particles(stars)
        self.petar_code.model_time = t_now
        self.petar_code.parameters.begin_time = t_now

        # rebuild Bridge if we have gas; otherwise just use PeTar directly
        if getattr(self, "gas_code", None) is not None:
            self._setup_bridge()
            self.bridge_code.add_system(self.petar_code, (self.gas_code,), False)
            self.bridge_code.add_system(self.gas_code)
            # if you later have an observer/tracker, re-add it here as well
            if self.track_background_gas_energy:
                self.bridge_code.add_system( self._gas_tracker )
            self.code = self.bridge_code
            self.code.model_time = self.model_time
            self.logger.info("[DCAF][SCALING] Bridge rebuilt after PeTar rescale")
        else:
            self.bridge_code = None
            self.code = self.petar_code


    # --- main loop ---------------------------------------------------------

    def evolve_model(self, t_end):
        """Advance the coupled system to t_end, interleaving outputs and
        formation events."""
        if self.model_time is None:
            raise RuntimeError("Call initialize_system() before evolve_model().")

        self.logger.info(f"[DCAF] Evolving to {t_end.in_(units.Myr)}")

        time = self.model_time
        t_output = self._ceil_to_block( time + self.dt_out )

        while time < t_end:
            # Next event times
            tnext = self.framework.get_next_formation_time()
            if tnext is None:
                tnext = 10*t_end
            tnext = self._ceil_to_block( tnext )

            # 0: finish; 1: output; 2: form stars
            event_times = (t_end, t_output, tnext)
            i_event, t_stop = min(enumerate(event_times), key=lambda x: x[1])
            self.logger.info(
                    f"[DCAF] (next event id: {i_event}) evolving from  "
                    f"{self.model_time.in_(units.Myr)} to "
                    f"{t_stop.in_(units.Myr)}"
                    )

            # Evolve dynamics up to t_stop if we actually need to advance time
            n_now = len(self.petar_code.particles)
            if (time < t_stop) and (n_now >= 2 or n_now == 0):
                # Evolve only if there is either a reasonable N or no stars (some codes allow)
                with self.logger.timing('[DCAF] Evolving *********************'):
                    self.code.evolve_model(t_stop)


            # Update clock
            time = t_stop
            self.model_time = time

            # Sync stellar evolution with petar

            if self.stellar_evolution and len(self.seba_code.particles) > 0:
                self.seba_code.evolve_model(self.model_time)

                # SeBa ->  target_stars
                seba_stars = self.seba_code.particles
                positions = get_key_positions(self.target_stars, seba_stars.key)
                target_seba_stars = self.target_stars[positions]

                target_seba_stars.mass = seba_stars.mass
                target_seba_stars.radius = seba_stars.radius
                target_seba_stars.stellar_type = seba_stars.stellar_type
                target_seba_stars.relative_age = seba_stars.relative_age
                target_seba_stars.relative_mass = seba_stars.relative_mass
                target_seba_stars.core_mass = seba_stars.core_mass
                target_seba_stars.COcore_mass = seba_stars.COcore_mass
                target_seba_stars.stellar_type = seba_stars.stellar_type

                # authoritative target_stars -> active PeTar particles
                code_positions = get_key_positions(self.petar_code.particles, seba_stars.key)
                petar_stars = self.petar_code.particles[code_positions]

                petar_stars.mass = target_seba_stars.mass
                petar_stars.radius = target_seba_stars.radius


            # 1) Output event
            if i_event == 1:
                self.write_output()
                t_output += self.dt_out

            # 2) Star-formation event
            if i_event == 2:
                with self.logger.timing('[GENERATE STARS]*********'):
                    new_stars = self.framework.form_stars(self.formed_stars)
                self._add_new_stars(new_stars)

            # check if we need to rescale
            if self.stars_per_worker is not None:
                nstars = len(self.petar_code.particles)
                #desired_workers = max((nstars - 1) // self.stars_per_worker + 1, 1)
                k = (nstars - 1) // self.stars_per_worker   # 0,1,2,...
                desired_workers = max(1 + k * self.workers_step, 1)
                if desired_workers > self._current_workers:
                    self.logger.info(
                        "[DCAF][SCALING] Rescaling PeTar workers from "
                        f"{self._current_workers} to {desired_workers} "
                        f"at N={nstars}"
                    )
                    self._rebuild_system(desired_workers)


    # --- I/O ---------------------------------------------------------------

    def write_output(self):
        with self.logger.timing('[WRITING OUTPUT]', False):
            self.logger.info(
                f"[DCAF] [WRITING OUTPUT] Snap: {self.__current_snapshot}, "
                f"Time {self.model_time.in_(units.Myr)} *************"
            )
            filename = os.path.join(self.output_folder, f"{self.snapshot_basename}{self.__current_snapshot:03d}")

            stars = self.formed_stars
            stars.collection_attributes.model_time = self.model_time
            stars.collection_attributes.code_time = (
                self.target_stars.collection_attributes.code_time
            )
            write_set_to_file(stars, filename + ".amuse", format='amuse')
            self.__current_snapshot += 1

            if self.framework.background_gas and hasattr(self.framework.background_gas, 'write_output'):
                self.framework.background_gas.write_output()

            self._energy_check()
            self._write_energy_row()

    # --- internals ---------------------------------------------------------

    def _add_new_stars(self, stars):
        with self.logger.timing('[ADDING STARS]', False):
            nactive = len(self.petar_code.particles)
            self.logger.info(f'[ADDING STARS]  Time: {self.model_time.value_in(units.Myr)}  '
                             f'{len(stars)} to {nactive}/{len(self.target_stars)} **************')

            # Add to PeTar
            #if len(self.petar_code.particles) == 0 :
            #    #no stars yet, lets add stars first and use the full code
            #    #potential energy
            #    self.petar_code.particles.add_particles(stars)
            #    self.__inject_new_stars_energy(stars,first_call = True)
            #else:
            #    # regular method, use petar potential method for new stars, 
            #    # and calculate 
            #    self.__inject_new_stars_energy(stars)
            #   


            if len(self.petar_code.particles) == 0 :
                U0 = 0 | units.J
            else:
                U0 = self.petar_code.potential_energy
            self.petar_code.particles.add_particles(stars)
            self.__inject_new_stars_energy(stars,U0=U0)

            if self.stellar_evolution:
                positions = get_key_positions(self.target_stars, stars.key)
                target_new_stars = self.target_stars[positions]

                target_new_stars.birth_time = self.model_time
                target_new_stars.initial_mass = target_new_stars.mass

                #sanity check
                if np.any(target_new_stars.in_seba):
                    raise ValueError(
                        "[DCAF][SEBA] Attempted to register stars already present in SeBa."
                    )

                target_new_stars.in_seba = True
                self.seba_code.particles.add_particles(target_new_stars.copy())


            for s in stars:
                self.logger.info(
                    f"[NEW_STAR] "
                    f"{self.model_time.value_in(units.Myr):.6f} "
                    f"{s.key:d} "
                    f"{s.mass.value_in(units.MSun):.6f} "
                    f"{s.x.value_in(units.parsec):.8f} "
                    f"{s.y.value_in(units.parsec):.8f} "
                    f"{s.z.value_in(units.parsec):.8f} "
                    f"{s.vx.value_in(units.kms):.6f} "
                    f"{s.vy.value_in(units.kms):.6f} "
                    f"{s.vz.value_in(units.kms):.6f}"
                )

    def __inject_new_stars_energy(self, new_stars, U0 = 0 | units.J):
        """
        Add the injected energy of `new_stars` to the budget
        """
        # --- kinetic of new stars
        v2 = new_stars.vx**2 + new_stars.vy**2 + new_stars.vz**2
        dE_kin = 0.5 * (new_stars.mass * v2).sum()

        dE_star = self.petar_code.potential_energy - U0

        # --- gas potential at new-star positions (assumed constant background here)
        if getattr(self, "gas_code", None) is not None:
            phi_gas = self.gas_code.get_potential_at_point(
                0 | units.m, new_stars.x, new_stars.y, new_stars.z
            )
            dE_gas = (new_stars.mass * phi_gas).sum()
        else:
            dE_gas = 0 | units.J

        dE = dE_kin + dE_star + dE_gas

        # add energy to the budget. Should be added to current and last, because
        # they should be equal
        self._total_energy[0] += dE
        self._total_energy[1] += dE
        # add to the individual energy budget for added stars
        self._energy_budgets[0] += dE

        if hasattr(self, "logger"):
            self.logger.debug(
                f"[ADD_STARS] dE_injected={dE.in_(units.J)}; "
                f"E_budget={self._total_energy[1].in_(units.J)}"
            )
        return dE


    def _ceil_to_block(self,t_si):
        """
        Get the closest time to a dt_soft multiple.
        We will perform operations only on those times for better performance
        """
        if self.dt_soft_eff is None:
            raise Exception( 'dt_soft_eff must be updated' )
        dtnb = self.converter.to_nbody(self.dt_soft_eff).number
        knb  = self.converter.to_nbody(t_si).number / dtnb
        k    = np.ceil(knb)                                 # nearest integer index
        return self.converter.to_si( ( k * dtnb)  | nbody_system.time )


    def _energy_check(self):
        first_check = False
        if self._last_energy_check is None:
            self._last_energy_check = self.model_time
            first_check = True

        stars = self.formed_stars
        # Ekin
        v2 = stars.vx**2 + stars.vy**2 + stars.vz**2
        Tstars = 0.5 * (stars.mass * v2).sum()

        # Estars_self (prefer solver-native)
        Ustars_self = self.petar_code.potential_energy

        # E_stars_gas (constant gas)
        if getattr(self, "gas_code", None) is not None:
            phi = self.gas_code.get_potential_at_point(0 | units.m, stars.x, stars.y, stars.z)
            U_stars_gas = (stars.mass * phi).sum()
            #if not np.is_finite(U_stars_gas):
                #print('not finite background gas energy',U_stars_gas)
                #raise Exception
        else:
            U_stars_gas = 0 | units.J

        #current total energy
        Etot = Tstars + Ustars_self + U_stars_gas

        # correcting: add work done by evolving gas potential (dPhi/dt term) ---
        W_gas = 0 | units.J  # work done by the gas
        if (getattr(self, "gas_code", None) is not None and 
            self._gas_tracker is not None ):
            #dt = self.model_time - self._last_energy_check
            #if dt > 0 | units.s:
            #    dphi_dt = self.gas_code.get_potential_derivative_at_point(
            #        stars.x, stars.y, stars.z
            #    )
            #    W_gas = (stars.mass * dphi_dt).sum() * dt

                # add only to the gas-evolution budget
                # (total energy itself is still T + U_self + U_bg)
            W_gas = self._gas_tracker.retrieve_stored_energy()
            self._energy_budgets[2] += W_gas
            #self._energy_components[3] = self.__energy_budgets[2]
            self._total_energy[1] += W_gas
            Etot += W_gas


        if first_check:
            # initial energy: we start with zero difference
            self._total_energy[0] = Etot 
            self._total_energy[1] = Etot

        self._total_energy[0] = self._total_energy[1] 
        self._total_energy[1] = Etot

        # numerical errors 
        eps          = np.abs(self._total_energy[1] - self._total_energy[0])
        Eref  = self._total_energy[1]
        relative_error = abs(eps / Eref) if Eref != (0 | units.J) else 0.0

        self._energy_components = [Tstars, Ustars_self, U_stars_gas, W_gas]
        self._energy_errors[0] = relative_error # step error
        self._energy_errors[1] += relative_error # cumulative error

        self._last_energy_check = self.model_time 

    def _write_energy_row(self):
        """
        Append to energy file
        """
        path = os.path.join(self.output_folder, "energy.dat")
        file_is_new = not os.path.exists(path)

        # extract scalars in SI (J) and time in Myr
        t_myr = float(self.model_time.value_in(units.Myr))
        t_nb = float( self.converter.to_nbody(self.model_time).number )
        vals = [
            float(self._total_energy[1].value_in(units.J)),  # Total energy
            float(self._energy_errors[0] ), # step error
            float(self._energy_errors[1] ), # cumulative error
            float(self._energy_components[0].value_in(units.J)),  # Ekin
            float(self._energy_components[1].value_in(units.J)),  # U_self
            float(self._energy_components[2].value_in(units.J)),  # U_bg
            float(self._energy_components[3].value_in(units.J)),  # W_gas
            float(self._energy_budgets[0].value_in(units.J)),  # E_new_stars
            float(self._energy_budgets[1].value_in(units.J)),  # E_SE
            float(self._energy_budgets[2].value_in(units.J)),  # E_gas_evol
            float(self._energy_budgets[3].value_in(units.J)),  # E_bin
        ]

        with open(path, "a") as f:
            if file_is_new:
                f.write(
                    "# units in Joules by default \n"
                    "# t_Myr t_nb E |dE/E|  sum(|dE/E|) T_*  U_*,*  U_*,gas"
                    "  W_gas  E_new_stars  E_SE   E_gas_evol  E_bin\n"
                )

            line = f"{t_myr:.6e}  {t_nb:.6e}  "
            for v in vals:
                line += f'{v:.6e}  '
            line += '\n'
            f.write(line)

    def _validate_formation_schedule(self,dt_soft_nb):
        """
        Validate that the intended formation schedule do not violate a set of
        rules designed to avoid adding stars twice during the same dt_soft
        block. Dcaf will be delay the formation events to the end of the next
        dt_soft block for better performance, but this means it can not add
        twice on the same block since may cause unintended results.
        Instead of handeling those events in dcaf, we leave the user to handle
        the decission on how to proceed. In case those rules are brokent, we
        raise an informative exception with suggested modifications to either
        the formation schedule or the chosen dt_soft.
        As a rule of thumb, formation events should be separated at least twice
        dt_soft in order to be consistent.

        The scheduled times should follow these rules:
          - First formation time t0 may be < dt_soft (we will place the first add at >= dt_soft).
          - From the first forming block onward, no two formation times may fall in the same dt_soft block.
        """
        dt_soft = self.converter.to_si(dt_soft_nb)
        ftimes = getattr(self.framework, "formation_times", None)
        if not ftimes or len(ftimes) == 0:
            raise ValueError("[DCAF][VALIDATOR] formation_times is empty.")

        ftimes = [ t.value_in(units.Myr) for t in ftimes ] | units.Myr

        t0 = ftimes[0]

        t_eff0 = dt_soft if t0 < dt_soft else self._ceil_to_block(t0)

        blocks = np.floor(((ftimes - t_eff0) / dt_soft)).astype(int)

        # Check duplicates after the first
        if len(np.unique(blocks[1:])) != len(blocks[1:]):
            # Gaps between consecutive events
            gaps = np.diff(np.array(ftimes))
            min_gap = np.min(gaps[1:]) if len(gaps) > 1 else None

            raise ValueError("\n"
                "[DCAF][VALIDATOR] Two or more formation events fall in the "
                "same dt_soft block. Counting from the first forming block at "
                f"{t_eff0.in_(units.Myr)} "
                f"(minimum width is dt_soft= {dt_soft.in_(units.Myr)})\n"
                f"Currently the minimum space between formation events provided"
                f" is {min_gap.in_(units.Myr) if min_gap else 'N/A'}. "
                " The solution is either decrease dt_soft below the minimum"
                " space between events or increase the space between events "
                " to a value greater than dt_soft. "
                " If using a class inherited from "
                "dcaf.framework.starformation.StarFormationFramework this means "
                "setting the dt_tolerance > dt_soft "
            )


class GasEnergyTracker:
    """
    This is a helper class to pass into Bridge in order to keep track of the
    work done by the gas at Bridge timestep level for energy conservation check.

    """
    def __init__(self, dcaf):
        self.dcaf = dcaf
        self.model_time = 0 | units.s   # Bridge reads this
        self.W_gas = 0 | units.J
        self._last_sum = None
        self._logfile = open('gas_energy_dbg.log','a')
    def get_gravity_at_point(self, *args, **kwargs):
        zero = 0 | units.ms**2
        return zero, zero, zero
    def get_potential_at_point(self, *args, **kwargs):
        return 0 | (units.kms**2)       # inert potential
    def stop(self, *args, **kwargs):
        pass
        return 
    # Called each Bridge substep
    def evolve_model(self, t_next):
        t_prev = self.model_time
        dt = t_next - t_prev
        stars = self.dcaf.petar_code.particles
        #print('here',type(stars),len(stars))
        #exit()
        if dt > 0 | units.s and getattr(self.dcaf, "gas_code", None) is not None:
            #x, y, z, m = stars.get_values_in_store(attributes=['x','y','z','mass'])

            dphi_dt = self.dcaf.gas_code.get_potential_derivative_at_point(
                stars.x, stars.y, stars.z
            )
            sum_now = (stars.mass * dphi_dt).sum()
            if self._last_sum is not None:
                self.W_gas += 0.5*( self._last_sum + sum_now  )*dt
            self._last_sum = sum_now

        self.model_time = t_next
        self._logfile.write(f'{t_next}   {self.W_gas}')

    def retrieve_stored_energy(self):
        wout = self.W_gas
        self.W_gas = 0 | units.J
        return wout

if __name__ == "__main__":
    pass
