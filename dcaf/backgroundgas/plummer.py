"""Implementation of an evolving Plummer sphere

This is a good example of how to inherit the
[BackgroundPotential][dcaf.backgroundgas.base.BackgroundPotential] class.

"""
from __future__ import annotations
from amuse.datamodel import Particles
from amuse.units import units
from amuse.units.quantities import Quantity, zero
from amuse.units.constants import G
from .base import BackgroundPotential
import numpy as np

class PlummerSphere(BackgroundPotential):
    r"""
    Plummer background potential with a mass decay and radial expansion.
    Used in
    [Farias et. al 2026](https://iopscience.iop.org/article/10.3847/1538-4357/ae8f41#Acknowledgment).


    ## Cloud evolution

    Before gas expulsion ($t \leq t_{\mathrm{ge}}$), the cloud may contract while
    its mass evolves as:

    $$
    a(t) = a_0 \sqrt{1 - \frac{t - t_0}{t_{\mathrm{col}}}},
    \quad
    M(t) = \max\left[0, M_0 + \dot{M}(t - t_0)\right].
    $$

    After gas expulsion ($t > t_{\mathrm{ge}}$), the mass remains fixed and the
    Plummer radius expands exponentially:

    $$
    a(t) = a(t_{\mathrm{ge}})
    \exp\left(\frac{t - t_{\mathrm{ge}}}{t_{\mathrm{exp}}}\right).
    $$

    Set `t_col = 0` to keep the cloud radius constant before gas expulsion.

    Args:
        mtot: Plummer Mass.
        rscale: Plummer scale length `a`.
        alpha_vir: Virial parameter alpha_vir = 2T/|W|;
            velocity dispersion scales as sqrt(alpha_vir).
        mdot: Scale mass depletion rate (implementing automatic).
        t0: Initial time of the model (important since temb = tge - t0).
        t_ge: Gas expulsion time.
        t_col: Cloud size collapse timescale (if zero then will stay constant).
        t_exp: Cloud size expansion timescale.

    Attributes:
        alpha_vir (float): Virial parameter used to scale the
            velocity dispersion.
        mdot (Quantity): Mass-change rate during the embedded
            phase.
        t_ge (Quantity): Transition time from embedded to
            expansion evolution.
        t_col (Quantity): Collapse timescale of the embedded
            phase.
        t_exp (Quantity): Expansion timescale after gas expulsion.
    """
    def __init__(
        self,
        mtot: Quantity,
        rscale: Quantity,
        alpha_vir: float = 1.0,
        mdot: Quantity = 0.0 | units.MSun / units.Myr,
        t0: Quantity = 0 | units.Myr,
        t_ge: Quantity = 3.0 | units.Myr,
        t_col: Quantity = 5.0 | units.Myr,
        t_exp: Quantity = 2 | units.Myr,
    ) -> None:
        super().__init__(mtot, rscale)
        self.alpha_vir = float(alpha_vir)
        self.mdot = mdot
        self.model_time = 0.0 | units.Myr
        self.t0 = t0
        self.t_ge = t_ge
        self.t_col = t_col
        self.t_exp = t_exp
        self.rscale_0 = self.rscale
        self.mtot_0 = self.mtot

        if self.t_ge < self.t0:
            raise ValueError("t_ge must be greater than or equal to t0.")

        embedded_duration = self.t_ge - self.t0
        if self.t_col < zero:
            raise ValueError("t_col must be non-negative.")

        if self.t_col > zero and self.t_col <= embedded_duration:
            raise ValueError(
                "t_col must exceed t_ge - t0, or be zero to disable collapse."
            )

    def get_potential_at_point(
        self,
        eps: Quantity,
        x: Quantity,
        y: Quantity,
        z: Quantity,
    ) -> Quantity:
        """Return the Plummer potential at one or more points.

        Args:
            eps: Gravitational softening length, unused by this model.
            x:  X coordinates.
            y:  Y coordinates.
            z:  Z coordinates.

        Returns:
            (Quantity): Gravitational potential at each point.
        """
        r2 = x**2 + y**2 + z**2
        a2 = self.rscale**2
        #print('time',self.model_time)
        #print('r2,a2,mtot',r2,a2,self.mtot)
        #print
        return - G * self.mtot / (r2+a2).sqrt()

    def get_gravity_at_point(
        self,
        eps: Quantity,
        x: Quantity,
        y: Quantity,
        z: Quantity,
    ) -> tuple[Quantity, Quantity, Quantity]:
        """Return the Plummer acceleration at one or more points.

        Args:
            eps: Gravitational softening length, unused by this model.
            x: X coordinates.
            y: Y coordinates.
            z: Z coordinates.

        Returns:
            (tuple[Quantity, Quantity, Quantity]): X, Y, and Z acceleration
            components.
        """
        r2 = x**2 + y**2 + z**2
        a2 = self.rscale**2
        denom32 = (r2 + a2)**1.5
        coef = - G * self.mtot / denom32
        ax = coef * x
        ay = coef * y
        az = coef * z
        return ax, ay, az

    def get_mass_inside_radius(self, r: Quantity) -> Quantity:
        """Return Plummer mass enclosed by radius `r`.

        Args:
            r: Radius or radii at which to evaluate enclosed mass.

        Returns:
            (Quantity): Enclosed gas mass.
        """
        return self.mtot * r**3 / (r**2 + self.rscale**2)**1.5

    def get_1d_velocity_dispersion_at_point(
        self,
        x: Quantity,
        y: Quantity,
        z: Quantity,
    ) -> Quantity:
        r"""
        From Heggie & Hut (2003):

        In virial equilibrium:

        $$
        \sigma_{\mathrm{1D}}^2(r) = \frac{G M}{6 a}
        \left(1 + \frac{r^2}{a^2}\right)^{-1/2}.
        $$

        We can scale the velocities to an arbitrary virial ratio by:

        $$
        \sigma_{\mathrm{1D}}^2(r, \alpha_{\mathrm{vir}})
        = \alpha_{\mathrm{vir}} \sigma_{\mathrm{1D}}^2(r),
        \qquad
        \sigma_{\mathrm{1D}}(r, \alpha_{\mathrm{vir}})
        = \sqrt{\alpha_{\mathrm{vir}}} \sigma_{\mathrm{1D}}(r).
        $$

        Args:
            x: X coordinates.
            y: Y coordinates.
            z: Z coordinates.

        Returns:
            (Quantity): Local one-dimensional velocity dispersion.
        """
        a = self.rscale
        a2 = a**2
        r2 = x**2 + y**2 + z**2
        sigma2 = (G * self.mtot) / (6.0 * a) * (1.0 + (r2/a2) )**(-0.5) * self.alpha_vir
        return sigma2.sqrt()

    def is_gas_relevant(self, stars: Particles,
                        radius_fraction: float = 0.9,
                        relevance_threshold: float = 0.999,
                        ) -> bool:
        """
        Return whether the gas is still dynamically relevant for the current
        simulation

        The default criterion compares the enclosed stellar mass fraction within
        the stellar 90% Lagrangian radius against a threshold. If the stars
        dominate the enclosed mass strongly enough, the gas is considered no
        longer relevant.

        Args:
            stars: Stellar particles used to measure the enclosed stellar mass
                fraction.
            radius_fraction: Fractional mass radius to use as reference.
            relevance_threshold: Stellar mass fraction inside `radius_fraction`
                above which gas is irrelevant.

        Returns:
            (bool): Whether the gas remains dynamically relevant.
        """
        if len(stars) == 0:
            return self.mtot > (0 | self.mtot.unit)

        mf = [float(radius_fraction)]
        rlag = stars.LagrangianRadii(mf=mf, cm=stars.center_of_mass())[0][0]

        mstar = stars.mass.sum()
        mgas = self.get_mass_inside_radius(rlag)
        mtot = mstar + mgas

        if mtot <= (0 | mtot.unit):
            return False

        stellar_fraction = mstar / mtot
        return stellar_fraction < relevance_threshold

    def evolve_model(self, tend: Quantity) -> None:
        r"""
        Advance to time `tend` using the two-phase background model:

        Embedded phase ($t \leq t_{\mathrm{ge}}$):

        $$
        a(t) = a_0 \sqrt{1 - \frac{t - t_0}{t_{\mathrm{col}}}},
        \qquad
        M(t) = \max\left[0, M_0 + \dot{M}(t - t_0)\right].
        $$

        Expansion phase ($t > t_{\mathrm{ge}}$):

        $$
        a(t) = a(t_{\mathrm{ge}})
        \exp\left(\frac{t - t_{\mathrm{ge}}}{t_{\mathrm{exp}}}\right),
        \qquad
        M(t) = M(t_{\mathrm{ge}}).
        $$

        Here, $\dot{M} < 0$ gives mass loss.

        Args:
            tend: Time to which the gas model is advanced.
        """
        # current to new time interval
        t = tend
        t0 = self.t0
        tge = self.t_ge
        tcol = self.t_col
        texp = self.t_exp

        # --- radius evolution ---
        if t <= tge:
            # collapse phase
            if tcol > zero:
                fac = 1.0 - (t - t0) / tcol
                fac = max(fac, 0.0)
                self.rscale = self.rscale_0 * fac**0.5
        else:
            # expansion phase
            # compute a at the break to ensure continuity
            if tcol == zero:
                a_break = self.rscale_0
            else:
                facb =  1.0 - (tge - t0)/tcol
                a_break = self.rscale_0 * facb**0.5
            self.rscale = a_break * np.exp((t - tge)/texp )

        # --- mass evolution ---
        if t <= tge:
            self.mtot = self.mtot_0 + self.mdot * (t - t0)
        else:
            # constant after break
            self.mtot = self.mtot_0 + self.mdot * (tge - t0)

        if self.mtot < (0.0 | self.mtot.unit):
            self.mtot = 0.0 | self.mtot.unit

        # update model clock
        self.model_time = t
        return

    def get_potential_derivative_at_point(
        self,
        x: Quantity,
        y: Quantity,
        z: Quantity,
    ) -> Quantity:
        r"""Return the analytic time derivative of the potential.

        At fixed position, define

        $$
        R = \sqrt{x^2 + y^2 + z^2 + a^2},
        \qquad
        \Phi = -\frac{G M}{R}.
        $$

        The time derivative is

        $$
        \frac{\partial \Phi}{\partial t}
        = -\frac{G \dot{M}}{R}
        + \frac{G M a \dot{a}}{R^3}.
        $$

        Args:
            x: X coordinates.
            y: Y coordinates.
            z: Z coordinates.

        Returns:
            (Quantity): `dPhi/dt` at each point.
        """
        r2 = x**2 + y**2 + z**2
        R  = (r2 + self.rscale**2).sqrt()

        zero_mass = 0 | self.mtot.unit
        zero_mdot = 0 | self.mdot.unit
        if self.model_time <= self.t_ge and self.mtot > zero_mass:
            Mdot = self.mdot
        else:
            Mdot = zero_mdot

        if self.model_time <= self.t_ge:
            if self.t_col > zero:
                adot = -0.5 * ( self.rscale_0**2 )  / self.rscale / self.t_col
            else:
                adot = 0 | units.pc/ units.Myr
        else:
            adot = self.rscale / self.t_exp


        term1 = -G*Mdot/R

        term2 =   G * self.mtot * self.rscale * adot / (R**3)

        return term1 + term2


class TabulatedPlummerSphere(PlummerSphere):
    """
    Plummer background potential where (mtot, rscale) are read from a time table.

    Expected table columns (floats) in fixed units:
        t   [Myr],  M [MSun],  a [pc]

    Default format: CSV with a header, e.g.\n
        t_myr,M_msun,a_pc
        0.0,1.0e5,1.0
        0.5,9.0e4,1.1
        1.0,7.5e4,1.3

    Args:
        mtot: Initial total mass used to initialize the parent
            Plummer model.
        rscale: Initial scale radius used to initialize the parent
            Plummer model.
        table_path: Path to the time, mass, and scale-radius table.
        alpha_vir: Virial parameter for velocity dispersion.
        clamp: Whether to clamp evaluations outside the table range.
        delimiter: Column delimiter used by the table.
        has_header: Whether the table's first row contains names.

    Notes:
        - Interpolation is linear in time.
        - If clamp=True (default): values are clamped to the first/last row outside the range.
          If clamp=False: out-of-range times raise ValueError.
        - Potential time-derivative uses piecewise-constant slopes from the bracketing interval.
    """

    def __init__(
        self,
        mtot: Quantity,
        rscale: Quantity,
        table_path: str,
        alpha_vir: float = 1.0,
        clamp: bool = True,
        delimiter: str = ",",
        has_header: bool = True,
    ) -> None:
        # Call parent but disable functional evolution parameters (mdot etc.)
        super().__init__(
            mtot, rscale,
            alpha_vir =float(alpha_vir),
            mdot = 0.0 | (units.MSun / units.Myr),
            t0 = 0 | units.Myr,
            t_ge = 0 | units.Myr,
            t_col = 0 | units.Myr,
            t_exp = 0 | units.Myr,
        )

        self.table_path = table_path
        self.clamp = bool(clamp)
        self.t0_table = None

        if has_header:
            tab = np.genfromtxt(table_path, delimiter=delimiter, names=True)
            cols = tab.dtype.names
            if (cols is None) or (len(cols) < 3):
                raise ValueError("TabulatedPlummerSphere: table must have at least 3 columns")
            t = np.array(tab[cols[0]], dtype=float)

            M = np.array(tab[cols[1]], dtype=float)
            a = np.array(tab[cols[2]], dtype=float)
        else:
            arr = np.loadtxt(table_path, delimiter=delimiter)
            if arr.ndim != 2 or arr.shape[1] < 3:
                raise ValueError("TabulatedPlummerSphere: table must have shape (N,>=3)")
            t, M, a = arr[:, 0].astype(float), arr[:, 1].astype(float), arr[:, 2].astype(float)

        if t.size < 2:
            raise ValueError("TabulatedPlummerSphere: table needs at least 2 rows")

        if np.any(~np.isfinite(t)) or np.any(~np.isfinite(M)) or np.any(~np.isfinite(a)):
            raise ValueError("TabulatedPlummerSphere: table contains non-finite values")

        if np.any(np.diff(t) <= 0):
            raise ValueError("TabulatedPlummerSphere: time column must be strictly increasing")

        t0_table = t[0]
        t = t - t0_table
        self.t0_table = float(t0_table)

        self._t_tab = t          # Myr (float)
        self._M_tab = M          # MSun (float)
        self._a_tab = a          # pc (float)

        # Initialise to t=0 (or clamp/raise based on table)
        self.evolve_model(self.model_time)

    def _bracket_index(self, t_myr: float) -> int:
        """
        Return i such that t_tab[i] <= t < t_tab[i+1].
        Assumes t is within [t0, tN]. Caller handles clamping/out-of-range.

        Args:
            t_myr: Time in Myr relative to the first table row.

        Returns:
            (int): Lower index of the bracketing table interval.
        """
        i = np.searchsorted(self._t_tab, t_myr, side="right") - 1
        if i < 0:
            i = 0
        if i >= self._t_tab.size - 1:
            i = self._t_tab.size - 2
        return i

    def _interp_params(self, t_myr: float) -> tuple[float, float]:
        """Interpolate mass and scale radius at a table time.

        Args:
            t_myr: Time in Myr relative to the first table row.

        Returns:
            (tuple[float, float]): Interpolated mass in solar
                masses and scale radius in parsecs.

        Raises:
            ValueError: If the time lies outside an unclamped table.
        """
        # handle out-of-range
        if t_myr <= self._t_tab[0]:
            if not self.clamp and (t_myr < self._t_tab[0]):
                raise ValueError("TabulatedPlummerSphere: time below table range")
            return self._M_tab[0], self._a_tab[0]
        if t_myr >= self._t_tab[-1]:
            if not self.clamp and (t_myr > self._t_tab[-1]):
                raise ValueError("TabulatedPlummerSphere: time above table range")
            return self._M_tab[-1], self._a_tab[-1]

        M = np.interp(t_myr, self._t_tab, self._M_tab)
        a = np.interp(t_myr, self._t_tab, self._a_tab)
        return M, a

    def _slopes_at_time(self, t_myr: float) -> tuple[float, float]:
        """
        Piecewise-constant slopes (dM/dt, da/dt) from the bracketing interval.
        If clamped outside range: slopes are 0.

        Args:
            t_myr: Time in Myr relative to the first table row.

        Returns:
            (tuple[float, float]): Mass and scale-radius slopes in solar masses
                per Myr and parsecs per Myr.

        Raises:
            ValueError: If the time lies outside an unclamped table.
        """
        if t_myr <= self._t_tab[0]:
            if not self.clamp and (t_myr < self._t_tab[0]):
                raise ValueError("TabulatedPlummerSphere: time below table range")
            return 0.0, 0.0
        if t_myr >= self._t_tab[-1]:
            if not self.clamp and (t_myr > self._t_tab[-1]):
                raise ValueError("TabulatedPlummerSphere: time above table range")
            return 0.0, 0.0

        i = self._bracket_index(t_myr)
        dt = self._t_tab[i+1] - self._t_tab[i]
        dMdt = (self._M_tab[i+1] - self._M_tab[i]) / dt
        dadt = (self._a_tab[i+1] - self._a_tab[i]) / dt
        return dMdt, dadt

    def evolve_model(self, tend: Quantity) -> None:
        """Set mass and scale radius from the table at `tend`.

        Args:
            tend:Target model time.

        Raises:
            ValueError: If the time lies outside an unclamped table.
        """
        # Set mtot and rscale from the table at time tend
        t_myr = tend.value_in(units.Myr)
        M, a = self._interp_params(t_myr)

        self.mtot = M | units.MSun
        self.rscale = a | units.pc

        self.model_time = tend
        return

    def get_potential_derivative_at_point(
        self,
        x: Quantity,
        y: Quantity,
        z: Quantity,
    ) -> Quantity:
        r"""
        dPhi/dt at point (x,y,z) for a time-evolving Plummer with tabulated M(t), a(t).

        $$
        R = \sqrt{x^2 + y^2 + z^2 + a^2},
        \qquad
        \Phi = -\frac{G M}{R},
        $$

        $$
        \frac{\partial \Phi}{\partial t}
        = -\frac{G \dot{M}}{R}
        + \frac{G M a \dot{a}}{R^3}.
        $$

        Args:
            x: X coordinates.
            y: Y coordinates.
            z: Z coordinates.

        Returns:
            (Quantity): `dPhi/dt` at each point.

        Raises:
            ValueError: If the model time lies outside an unclamped table.
        """
        r2 = x**2 + y**2 + z**2
        R  = (r2 + self.rscale**2).sqrt()

        t_myr = self.model_time.value_in(units.Myr)
        dMdt, dadt = self._slopes_at_time(t_myr)

        Mdot = (dMdt | units.MSun) / units.Myr
        adot = (dadt | units.pc) / units.Myr

        term1 = -G * Mdot / R
        term2 =  G * self.mtot * self.rscale * adot / (R**3)
        return term1 + term2

def test_plummer_evolution(
    mtot: Quantity = 1e5 | units.MSun,
    rscale: Quantity = 1 | units.pc,
    mdot: Quantity = -1e4 | (units.MSun / units.Myr),
    t0: Quantity = 0 | units.Myr,
    t_ge: Quantity = 3 | units.Myr,
    t_col: Quantity = 5 | units.Myr,
    t_exp: Quantity = 2 | units.Myr,
    model: PlummerSphere = None,
    times: Quantity = [0, 1, 2, 3, 4, 6, 8] | units.Myr,
    test_position: str | tuple[Quantity, Quantity, Quantity] = "auto",
) -> None:
    """
    Test the evolution of the plummer model.
    model is an instance of the cloud, if given, then the parameters are ignored
    the the model instance is used instead.

    Args:
        mtot: Initial gas mass for a newly created model.
        rscale: Initial scale radius for a newly created model.
        mdot: Embedded-phase mass-change rate.
        t0: Initial model time.
        t_ge: Gas-expulsion time.
        t_col: Collapse timescale.
        t_exp: Expansion timescale.
        model: Existing model to evaluate instead of creating one.
        times: Times at which to print diagnostic values.
        test_position: `"auto"` or an `(x, y, z)` position.
    """

    if model is None:
        model = PlummerSphere(
            mtot=mtot,
            rscale=rscale,
            mdot=mdot,
            t0=t0,
            t_ge=t_ge,
            t_col=t_col,
            t_exp=t_exp,
            alpha_vir=1.0
        )

    # --- test times and point ---
    if test_position == 'auto':
        xtest = 0|units.pc
        ytest = 0|units.pc
        ztest = model.rscale
    else:
        xtest,ytest,ztest = test_position

    print(f"{'t [Myr]':>8} {'a [pc]':>10} {'M [Msun]':>12} "
          f"{'Phi(r)':>9} {'dPhi/dt':>15}")
    print("-" * 60)
    for t in times:
        model.evolve_model(t)
        phi = model.get_potential_at_point(0|units.pc,xtest,ytest,ztest)
        phidot = model.get_potential_derivative_at_point(xtest,ytest,ztest)

        print(f"{t.value_in(units.Myr):8.2f} "
              f"{model.rscale.value_in(units.pc):10.3f} "
              f"{model.mtot.value_in(units.MSun):12.1f} "
              f"{phi.value_in(units.kms**2):12.3e}"
              f"{phidot.value_in(units.kms**2/units.Myr):12.3e}")

if __name__ == "__main__":
    test_plummer_evolution()
