from math import isfinite
import numpy as np
from amuse.units import units
from amuse.datamodel.particles import Particles
#pop would do, but this is more efficient
from collections import deque

class StarFormationFramework :
    """ Star Formation Framework 

    This class handles the decision making during the star formation process. It
    includes how fast stars form (star formation rate), what are the initial
    phase-space coordinates of the new stars stars, what are the new phase-space
    coordinates of those stars and how the background gas potential changes as
    new stars form (e.g. via star formation efficiency) and any other more
    complex formation mechanism the user may want to include.

    Input: 
        target_stars : amuse.Particles - The list of stars that will be added to
            the simulation. The order of appearance will be the same as this
            set.

        star_formation_rate : The star formation rate at which new stars will
            appear. If is an amuse.quantity (in e.g. Msun/Myr) the value will be
            constant through the simulation.
            If star_formation_rate == 'infty', all stars will be added at the
            beginning of the simulation.
            If a (time,star_formation_rate) tuple is provided, then the instant
            star formation rate will be linearly interpolated over time within
            the range provided by the table (note that time and
            star_formation_rate should be amuse quantities'

        nstart : int, The number of stars to form initially. The code will
            advance the necessary time, derived from the star formation rate, in
            order to reach the required mass to form these initial stars.

        background_gas : a dcaf.backgroundgas.BackgroundPotential inherited
            class. This class controls how the background gas evolves and react
            to the formation of the new stars and provide the (additional)
            acceleration felt by the star particles during the simulation.


    The typical workflow will be by the User creating an inherited class using
    StarFormationFramework as parent class and overwriting the appropiate
    functions.

    Stars will be added to the N-body simulation provided a star_formation_rate
    value (function or table) and the form_stars method will be called at the
    specific times where a new star is scheduled.

    The typical method an User may want to redefine is:

    StarFormationFramework.form_stars : Whenever new stars are shceduled to form
        this is the function that will provide the new amuse.Particles to be
        added. An external generator of stars can be placed here for finer
        tuning on the coordinates of the new stas (for instance, based on the
        positions of existing stars).


    The background gas model should be defined by an inherited class from the
    dcaf.bakgroundgas.BackgroundPotential that contain the necessary methods to
    provide the potential and acceleration required by the bridge scheme.

    This class can also be used for decision making at the generation of stars,
    for instance, any BackgroundPotential inherited class should have defined
    methods like: 
        get_1d_velocity_dispersion_at_point : to obtain the velocity
            dispersion at the position of the new star if we want the new star
            to inherit the kinematics of the parent cloud. 
    Or,
        get_mass_inside_radius : to obtain the enclosed mass at the new particle
            position in case we want that information instead.

    Note that background_gas will be passed to the Bridge scheme to handle the
        gas to stars interactionn, therefore must have defined the appropriate
        get_gravity_at_point
    """

    def __init__( self, target_stars, star_formation_rate = 'infty',
                 nstart = 2, background_gas = None ,
                 dt_tolerance = 1e-3 | units.Myr):
        self.target_stars = target_stars
        self.star_formation_rate = star_formation_rate
        self.nstart = nstart
        self.background_gas = background_gas
        self.dt_tolerance = dt_tolerance

        # initialize the is_active flag. We will use this to distinguish stars
        # that are already formed.
        self.target_stars.is_active = False

        self.schedule_formation()

    def get_next_formation_time(self):
        return self.__next_formation_time

    def get_last_formation_time(self):
        ## useful for restart
        next_t = self.get_next_formation_time()

        if next_t is None:
            return None

        if len(self.formation_times) == 0:
            return next_t

        return self.formation_times[-1]


    def extract_next_event(self):
        """
        Retrieve next scheduled stars and setup the next formation event.
        This function should be called by form_stars to obtain new stars to
        form.
        NOTE: we are also flaggin stars as active here. SO this function MUST be
        called by form_stars.
        """
        #retrieve the new stars and forward framework time to current time
        new_stars = self.__next_stars

        if new_stars is not None and len(new_stars) > 0:
            active_mask = np.isin(self.target_stars.key, new_stars.key)
            self.target_stars[active_mask].is_active = True

        self.model_time = self.__next_formation_time
        self.__setup_next_event()

        return new_stars

    def schedule_formation(self, t0=0 | units.Myr):
        """
        Build a schedule of star formation in batches.

        Rules
        -----
        - If `system_id` exists, stars sharing the same positive `system_id`
          are always added together.
        - If `system_id` is missing, or `system_id <= 0`, the star is treated
          as a singleton.
        - Order is the order of first appearance in `target_stars`.
        - `nstart` is a soft lower bound in number of stars: whole systems are
          added until the first batch reaches or exceeds `nstart`.
        - Systems whose formation times differ by less than `dt_tolerance`
          are merged into the same formation event.
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

    def __setup_next_event(self):
        print('settingup',len(self.formation_sequence) )
        if len(self.formation_sequence) > 0:
            newstars = self.formation_sequence.popleft()
            formation_time = self.formation_times.popleft()
            self.__next_formation_time = formation_time
            self.__next_stars = newstars
        else:
            self.__next_formation_time = None
            self.__next_stars = None

    def get_velocity_dispersion_at_point(self,x,y,z):
        if self.background_gas:
            return self.background_gas.get_1d_velocity_dispersion_at_point(x,y,z)
        else:
            # fallback to current stars velocity dispersion
            vx = self.target_stars.vx.std()
            vy = self.target_stars.vy.std()
            vz = self.target_stars.vz.std()
            return np.mean([vx,vy,vz])

    def form_stars(self,active_stars=Particles()):
        """
        Retrieve the new stars applying the formation rules and schedule
        the next formation event.

        By default the new stars are passed directly from the scheduled stars,
        i.e. with the positions and velocities from the original particle list.

        If a more complex formation scenario is needed, for instance using the
        formation rules from dcaf.factory.distance_based, then this function
        should be overwritten.

        Note that the first call of this function will be done with an empty set
        of active_stars, then it should handle such case.

        Example:

        Here is a basic example using the function generate_stars from
        dcaf.factory.distance_based (see doc).
        It generate new positions [and velocities?] for the number of requested
        stars based on the position of the existing set and a predefined PDF of
        closest neighbours (see REFERENCE).

        Note that the function MUST handle EMPTY active_stars and the
        generate_stars function MUST handle n_new == 0

        In this example, if active_stars is empty will just return the schedule
        stars with their original coordinates on a gradual formation simulation
        with NO GAS background.
        If n_new== 0 new_stars is an empty set of Particles

        from dcaf.factory import distance_based 
        from dcaf.dcaf import DcafSystem

        class MyFormationFramework(StarFormationFramework):

            def form_stars(self,active_stars):
                # Get the next scheduled stars to form. This method also prepare
                # the next event for the next extract_next_event call.

                next_stars = self.extract_next_event()
                n_new = len(next_stars)
                
                # Obtain new positions based on the existing stars
                # Note that generate_stars should handle n_new == 0
                 The first call will be done with an empty
                # active_stars and such case must be handled here.

                if len(active_stars) ==  0:
                    new_stars = next_stars # first time, keep original
                    coordinates
                else:
                    new_stars = distance_based.generate_stars( active_stars, n_new )
                
                if len(new_stars) > 0 :
                    new_stars.mass = next_stars.mass

                return new_stars

        # Setup the final stars
        ntot = 1000
        Rpl = 10 |units.pc
        masses = new_kroupa_mass_distribution(ntot)
        target_stars =  new_plummer_model(ntot)
        target_stars.mass = masses

        # Setup the final time and the star formation rate as constant
        tend = 10 | units.Myr
        star_formation_rate = masses.sum() / tend

        framework = MyFormationFramework(target_stars,star_formation_rate = star_formation_rate)
        
        # run with default code configuration 
        system = DcafSystem( framework )

        """

        new_stars = self.extract_next_event()

        return new_stars
