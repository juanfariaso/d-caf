"""Turbulent Core Model initial conditions for stellar clusters.

The implementation follows the Turbulent Core Model of
[McKee & Tan (2003)](https://iopscience.iop.org/article/10.1086/346149), as
applied to cluster formation by
[Farias et al.
(2017)](https://iopscience.iop.org/article/10.3847/1538-4357/aa63f6) series.

Physical inputs and generated particle attributes use AMUSE quantities.
"""
import numpy as np
from dataclasses import dataclass, field

from amuse.lab import units, Particles, new_kroupa_mass_distribution
from amuse.units.quantities import Quantity
from scipy.special import hyp2f1


@dataclass
class TurbulentCoreParams:
    """Calculate and store parameters of the Turbulent Core Model.

    Class that compute and contain the parameters of the TCM.
    See Mckee & Tan 2003 and Farias et al. 2017

    Args:
        Mc: Total mass of the turbulent gas cloud.
        k_rho: Radial density slope. Default: 1.5.
        alpha_vir: Virial parameter of the cloud.
        phi_Pc: Core-pressure factor.
        phi_B: Magnetic-field factor.
        surface_density: Cloud surface density.
        fg: Gas mass fraction entering the TCM factors.
        aspect_ratio: z-stretch (>0), 1.0 -> spherical (Default).
        ar_velocity_scale: if True, apply aspect-ratio velocity scaling.
            Default: True.

    Attributes:
        Rc: Cloud radius.
        sigma_surf: Surface 1D velocity dispersion.
        sigma_1d: Global 1D velocity dispersion.
        phi_Pmean: Dimensionless pressure factor.
        phi_geom: Geometric factor for aspect ratio.
    """
    # inputs
    Mc: Quantity = field(default_factory=lambda: 3000 | units.MSun)
    k_rho : float = 1.5
    alpha_vir : float = 1.0
    phi_Pc : float = 2.0
    phi_B : float = 2.8
    surface_density: Quantity = field(
        default_factory=lambda: 0.1 | (units.g * units.cm**-2)
    )
    fg : float = 1.0
    aspect_ratio : float = 1.0
    ar_velocity_scale : bool = True

    # derived 
    Rc: Quantity = field(init=False)
    sigma_surf: Quantity = field(init=False)
    sigma_1d: Quantity = field(init=False)
    phi_Pmean: float = field(init=False)
    phi_geom: float = field(init=False)

    def _phi_geom(self, aspect_ratio: float) -> float:
        if aspect_ratio <= 0:
            raise ValueError("aspect_ratio must be > 0")
        if aspect_ratio == 1.0:
            return 1.0
        phi_geom = (
            float(hyp2f1(0.5, -0.25, 1.5, 1.0 - aspect_ratio ** -2) * 
                  aspect_ratio ** 0.5)
            )
        return phi_geom

    ## Derive parameters
    def __post_init__(self):
        phi_geom = self._phi_geom(self.aspect_ratio)
        kP = 2.0 * (self.k_rho - 1.0)
        A = (3.0 - self.k_rho) * (self.k_rho - 1.0) * self.fg
        self.phi_Pmean = float((3.0 * np.pi / 20.0) * self.fg * self.phi_B * self.alpha_vir * phi_geom)

        # Surface velocity dispersion
        b = (self.phi_Pc * self.phi_Pmean / A / (kP ** 2) / (self.phi_B ** 4)) ** (1.0 / 8.0)
        c = (self.Mc / (3000.0 | units.MSun)) ** 0.25
        d = (self.surface_density.value_in( units.g * units.cm ** -2) ) ** 0.25
        sigma_surf = (5.08 * b * c * d) | units.kms

        # Core radius
        #aR = (A / (kP ** 2) / self.phi_Pc / self.phi_Pmean) ** (1/4. )
        #bR = (self.Mc / (60 | units.MSun)) ** (1./2)
        #cR = (self.surface_density / (1 | units.g * units.cm ** -2)) ** (-1/2.)
        #Rc = (0.071 * aR * bR * cR) | units.parsec

        aR = float((A / (kP**2) / self.phi_Pc / self.phi_Pmean)**0.25)
        bR = float((self.Mc / (60 | units.MSun)))**0.5
        cR = float(self.surface_density.value_in(units.g * units.cm**-2) )**(-0.5)

        Rc = (0.071 * aR * bR * cR) | units.parsec

        # Volume-weighted (global) 1D sigma from surface scaling
        sigma_1d = (2.0 * (3.0 - self.k_rho) / (8.0 - 3.0 * self.k_rho) * sigma_surf).in_(units.kms)

        if self.aspect_ratio != 1.0 and self.ar_velocity_scale:
            vscale = np.sqrt(
                np.arcsinh(self.aspect_ratio ** 2 - 1.0) / (self.aspect_ratio ** 2 - 1.0) * phi_geom
            )
            sigma_surf *= np.sqrt(vscale)
            sigma_1d *= np.sqrt(vscale)

        # assign
        self.Rc = Rc
        self.sigma_surf = sigma_surf
        self.sigma_1d = sigma_1d
        self.phi_geom = float(phi_geom)


def make_turbulent_core_cluster(
    Mc: Quantity = 3000 | units.MSun,
    sfe: float = 0.5,
    k_rho: float = 1.5,
    alpha_vir: float = 1.0,
    phi_Pc: float = 2.0,
    phi_B: float = 2.8,
    surface_density: Quantity = 0.1 | units.g * units.cm ** -2,
    fg: float = 1.0,
    aspect_ratio: float = 1.0,
    keps: float = -1.0,
    masses: Quantity = None,
    nstars: int = None,
    m_equal: Quantity = None,
    ar_velocity_scale: bool = True,
    seed: int = 432,
) -> tuple[Particles, TurbulentCoreParams, float]:
    """Create a single-stars cluster realization from the Turbulent Core Model.

    Create a single-stars cluster realisation with positions and velocities
    consistent with the turbulent-core model.

    You can provide either:
      - `masses`: a Quantity array of stellar masses [MSun]; or
      - `nstars` AND `m_equal`: use equal-mass stars of mass `m_equal`.

    Args:
        Mc: Total gas-cloud mass.
        sfe: Retained input parameter. It does not currently change the
            generated stellar masses or positions.
        k_rho: Radial density slope.
        alpha_vir: Virial parameter passed to ``TurbulentCoreParams``.
        phi_Pc: Core-pressure factor passed to ``TurbulentCoreParams``.
        phi_B: Magnetic-field factor passed to ``TurbulentCoreParams``.
        surface_density: Cloud surface density passed to ``TurbulentCoreParams``.
        fg: Gas mass fraction passed to ``TurbulentCoreParams``.
        aspect_ratio: z-axis stretch of the generated spatial distribution.
        keps: Exponent parameter controlling the cumulative-mass radial sampling
            relation.
        masses: Stellar mass array. Provide this or both ``nstars`` and
            ``m_equal``.
        nstars: Number of equal-mass stars to create when ``masses`` is omitted.
        m_equal: Mass assigned to each star when ``masses`` is omitted.
        ar_velocity_scale: Whether to apply the aspect-ratio velocity scaling.
        seed: Seed for the NumPy random generator.

    Returns:
        (Particles): Stars with mass, position, velocity, and
            radius attributes.
        (TurbulentCoreParams): Derived TCM cloud parameters.
        (float): Realized stellar mass fraction ``sum(stars.mass) / Mc``.

    Raises:
        ValueError: If neither a mass array nor both equal-mass inputs are
            provided, or if ``k_rho`` and ``keps`` make the radial sampling exponent
            undefined.
    """
    if masses is None:
        if nstars is None or m_equal is None:
            raise ValueError("Provide either `masses` or both `nstars` and `m_equal`.")
        masses = (np.ones(nstars) | units.none) * m_equal
    else:
        masses = masses.in_(units.MSun)

    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    params = TurbulentCoreParams(
        Mc=Mc,
        k_rho=k_rho,
        alpha_vir=alpha_vir,
        phi_Pc=phi_Pc,
        phi_B=phi_B,
        surface_density=surface_density,
        fg=fg,
        aspect_ratio=aspect_ratio,
        ar_velocity_scale=ar_velocity_scale,
    )

    Mstars_eff = masses.sum().in_(units.MSun)
    sfe_eff = (Mstars_eff / Mc)

    n = len(masses)

    # Generalised exponent (keps = -1 -> 1/(3-k_rho))
    den = 6.0 - k_rho * (keps + 3.0)
    if np.isclose(den, 0.0):
        raise ValueError(f"Invalid (k_rho, keps) combination: 6 - k_rho*(keps+3) == 0 (k_rho={k_rho}, keps={keps})")
    expo = 2.0 / den

    # Randomise order to avoid spatial correlations with mass unless wanted
    order = np.arange(n)
    rng.shuffle(order)
    m_sorted = masses[order]

    cmass = np.cumsum(m_sorted.value_in(units.MSun)) | units.MSun
    ratio = (cmass / Mstars_eff)

    r = params.Rc.value_in(units.parsec) * ( ratio ** expo ) 

    r = r | units.parsec

    # Isotropic angles
    u = rng.uniform(size=n)
    costheta = 2.0 * u - 1.0
    sintheta = np.sqrt(1.0 - costheta ** 2)
    phi = 2.0 * np.pi * rng.uniform(size=n)

    x = r * sintheta * np.cos(phi)
    y = r * sintheta * np.sin(phi)
    z = r * costheta
    if aspect_ratio != 1.0:
        z *= aspect_ratio

    # Velocity dispersion scaling: sigma(r) = sigma_surf * (r/Rc)^{(2-k_rho)/2}
    s1d = (
            params.sigma_surf * (r / params.Rc) ** ((2.0 - k_rho) / 2.0)
        ).in_(units.kms)

    vx = (rng.normal(scale=1.0, size=n) | units.none) * s1d
    vy = (rng.normal(scale=1.0, size=n) | units.none) * s1d
    vz = (rng.normal(scale=1.0, size=n) | units.none) * s1d

    stars = Particles(n)
    inv = np.argsort(order)
    stars.mass = m_sorted[inv]
    stars.x = x[inv]
    stars.y = y[inv]
    stars.z = z[inv]
    stars.vx = vx[inv]
    stars.vy = vy[inv]
    stars.vz = vz[inv]
    stars.radius = (0.01 | units.parsec)

    return stars, params, float(sfe_eff)


# -----------------------
# Simple test helper (optional)
# -----------------------

# TODO: Check for another way of doing this. Ideally it should be set at runtime
# following the current sfe_ff. All stars at once should be set with an 
# sfe_ff == infty

def make_kroupa_masses(
    Mstars: Quantity,
    mmin: Quantity = 0.01 | units.MSun,
    mmax: Quantity = 100 | units.MSun,
) -> Quantity:
    """Draw Kroupa IMF masses up to a target stellar mass.

    Args:
        Mstars: Target total stellar mass.
        mmin: Lower stellar-mass limit for the Kroupa IMF.
        mmax: Upper stellar-mass limit for the Kroupa IMF.

    Returns:
        (Quantity): One-dimensional stellar masses in solar masses. The final
            draw is discarded when it would exceed ``Mstars``, so the returned
            sum does not exceed the target.
    """
    mtot = 0 | units.MSun
    out = []
    while mtot < Mstars:
        m = new_kroupa_mass_distribution(1, mass_min=mmin, mass_max=mmax)[0]
        out.append(m.value_in(units.MSun))
        mtot += m
    masses = (np.array(out) | units.MSun)
    if masses.sum() > Mstars and len(masses) > 0:
        masses = masses[:-1]
    return masses

# test_tc_core.py
"""
Quick executable test: create a cluster and plot XY positions coloured by speed,
plus a radial sigma profile vs. the expected scaling.

Usage:
    python test_tc_core.py
"""
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    Mc = 3000 | units.MSun
    sfe = 0.5
    k_rho = 1.5
    aspect_ratio = 1.0

    Mstars = sfe * Mc
    masses = make_kroupa_masses(Mstars, mmax=100 | units.MSun)

    stars, params, sfe_eff = make_turbulent_core_cluster(
        Mc=Mc,
        sfe=sfe,
        k_rho=k_rho,
        surface_density=0.1 | units.g * units.cm ** -2,
        alpha_vir=1.0,
        phi_Pc=2.0,
        phi_B=2.8,
        fg=1.0,
        aspect_ratio=aspect_ratio,
        keps=-1.0,
        masses=masses,
        seed=432,
    )

    print(f"effective SFE realised: {sfe_eff:.3f}")
    r = (stars.position.lengths() / params.Rc)
    speed = stars.velocity.lengths().value_in(units.kms)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    sc = plt.scatter(
        stars.x.value_in(units.parsec),
        stars.y.value_in(units.parsec),
        c=speed,
        s=6,
        alpha=0.8,
    )
    plt.xlabel("x [pc]")
    plt.ylabel("y [pc]")
    plt.title("Turbulent-core XY positions (colour: |v| km/s)")
    cbar = plt.colorbar(sc)
    cbar.set_label("velocity [km/s]")
    plt.axis("equal")

    plt.subplot(1, 2, 2)
    bins = np.linspace(0, 1.2, 20)
    which = np.digitize(r, bins)
    sig_obs = []
    rmid = []
    for b in range(1, len(bins)):
        sel = which == b
        if sel.any():
            vx = stars[sel].vx.value_in(units.kms)
            vy = stars[sel].vy.value_in(units.kms)
            vz = stars[sel].vz.value_in(units.kms)
            sig_obs.append( np.sqrt(np.var(vx) + np.var(vy) + np.var(vz)) / np.sqrt(3.0) )
            rmid.append(0.5 * (bins[b] + bins[b - 1]))

    rmid = np.array(rmid)
    sig_obs = np.array(sig_obs)

    rplot = np.linspace(1e-3, 1.2, 200)
    sig_exp = (params.sigma_surf * (rplot ) ** ((2.0 - k_rho) / 2.0)).value_in(units.kms)

    plt.plot(rplot, sig_exp, lw=2, label="expected σ(r)")
    if len(rmid) > 0:
        plt.scatter(rmid, sig_obs, s=18, label="measured (binned)")
    plt.xlabel("r / Rc")
    plt.ylabel("σ [km/s]")
    plt.title("Velocity dispersion profile")
    plt.legend()
    plt.tight_layout()
    plt.savefig('turbulent_clump_model_test.pdf')


# -----------------------
# Pytest numerical check
# -----------------------
def test_sigma_slope_matches():
    Mc = 1000 | units.MSun
    sfe = 0.5
    k_rho = 1.5
    masses = make_kroupa_masses(sfe * Mc, mmax=50 | units.MSun)
    stars, params, _ = sample_turbulent_core_cluster(Mc=Mc, sfe=sfe, k_rho=k_rho, masses=masses, seed=42)
    r = (stars.position.lengths() / params.Rc)
    mask = r > 1e-2
    sig = stars.velocity.lengths().value_in(units.kms)
    coeffs = np.polyfit(np.log10(r[mask]), np.log10(sig[mask]), 1)
    slope_measured = coeffs[0]
    slope_theory = (2.0 - k_rho) / 2.0
    assert np.isclose(slope_measured, slope_theory, atol=0.2)
