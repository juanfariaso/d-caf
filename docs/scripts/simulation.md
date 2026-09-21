# Parametric Plummer Simulation

`dcaf.scripts.parametric_plummer` is the supplied end-to-end runner for a
parametric star-cluster formation simulation. It samples a Kroupa stellar mass
distribution, creates a Plummer realization, applies a radial velocity-
dispersion profile, schedules gradual star formation, and evolves the result
with D-CAF in an evolving Plummer background potential.

This is a specific workflow, rather than a general interface for every
`StarFormationFramework` or background-gas model. For custom Python workflows,
see the [DCAF system API](../reference/dcaf/dcaf.md) and the
[star-formation framework API](../reference/dcaf/framework/starformation.md).

## Run A Simulation

Install D-CAF first using the [installation guide](../install.md). This provides
the `dcaf-plummer` command.

Run the module from a working directory where its output files should be
created:

```shell
dcaf-plummer --config config.yaml
```

The runner writes the effective parameters to `config.yaml` in the current
working directory before starting. Use a separate working directory for each
independent run if you want to keep configurations and gas-output files
separate.

Before committing resources to a simulation, confirm that the configuration is
accepted without initializing D-CAF:

```shell
dcaf-plummer --config config.yaml --dry_run
```

The `--dry_run` option checks parsing only. It does not construct particles,
validate the physical model, or initialize PeTar.

## Configuration

The runner starts from its built-in defaults, applies values from `--config`,
then applies command-line options. A command-line value therefore overrides the
same YAML value.

Quantities in YAML use a number followed by the expected unit, for example
`10.0 parsec` or `30.0 Myr`.

### Physical Model

The runner treats `Mstars` as the target final stellar mass and `sfe` as the
global star-formation efficiency. It therefore derives the cloud mass as

$$
M_{\rm cl} = \frac{M_\star}{\epsilon}.
$$

`Rcl` defines the cloud radius used to calculate the mean-density free-fall
time,

$$
t_{\rm ff} = \left(\frac{3\pi}{32 G \rho_{\rm mean}}\right)^{1/2},
\qquad
\rho_{\rm mean} = \frac{3 M_{\rm cl}}{4\pi R_{\rm cl}^3}.
$$

The Plummer background mass is normalized so that the mass enclosed within
`Rcl` equals `Mcl`. With gas scale radius `Rpl`, the enclosed fraction is

$$
f_{\rm Pl} = \frac{R_{\rm cl}^3}
{\left(R_{\rm cl}^2 + R_{\rm Pl}^2\right)^{3/2}},
\qquad
M_{\rm Pl} = \frac{M_{\rm cl}}{f_{\rm Pl}}.
$$

Stars form at a constant scheduled rate throughout the embedded phase,

$$
\dot{M}_\star = \frac{M_\star}{t_{\rm ge}}.
$$

During that same phase, `mdot_factor` sets an optional gas mass change relative
to this stellar formation rate:

$$
\dot{M}_{\rm gas} = \texttt{mdot_factor}\,\dot{M}_\star,
\qquad
M_{\rm gas}(t) = M_{\rm Pl} + \dot{M}_{\rm gas} t
\quad (0 \leq t \leq t_{\rm ge}).
$$

Thus the default `mdot_factor: 0` keeps the gas mass fixed. At `t_ge`, gas mass
growth or depletion stops and the Plummer scale radius expands exponentially on
`t_exp`:

$$
R_{\rm Pl}(t) = R_{\rm Pl}(t_{\rm ge})
\exp\left(\frac{t-t_{\rm ge}}{t_{\rm exp}}\right).
$$

`Fmax` is the maximum fraction of the initial gas mass enclosed within 10% of
the cloud's initial mass radius during the embedded-phase collapse. It sets the
pre-expulsion collapse timescale; leave it `null` to keep the gas scale radius
fixed during the embedded phase.

```yaml
seed_index: 0
Rcl: 10.0 parsec
Rpl: 7.0 parsec
Mstars: 2000.0 MSun
sfe: 0.3
t_end: 30.0 Myr
dt_out: 0.05 Myr
nworkers: 2
field_binaries: false
```

These are the possible parameters accepted in `config.yaml`:

| Key | Unit | Default | Meaning |
| --- | --- | --- | --- |
| `seed_index` | -- | `0` | Index into the runner's fixed list of random seeds. |
| `tff` | Myr | `null` | Target cloud free-fall time. When given, it determines `Rcl` while preserving `Rpl / Rcl`. |
| `Rcl` | pc | `10.0` | Cloud radius used when `tff` is not given. |
| `Rpl` | pc | `7.0` | Initial gas Plummer scale radius. |
| `Mstars` | MSun | `2000.0` | Target stellar mass used for IMF sampling. |
| `sfe` | -- | `0.3` | Target star formation efficiency used to derive the cloud mass: `Mcl = Mstars / sfe`. |
| `Fmax` | -- | `null` | Optional maximum collapse fraction, the fraction of mass that the interior 10% cloud mass radius raise before gas-expulsion, used to derive `t_col`.  |
| `tge_over_tff` | -- | `1.0` | Gas-expulsion time (when gas expulsion begins) in free-fall-time units, unless `t_ge` is given explicitly. |
| `texp_over_tff` | -- | `1.0` | Gas-expansion timescale in free-fall-time units, unless `t_exp` is given. |
| `t_ge` | Myr | `null` | Time when gas-expulsion begins. |
| `t_exp` | Myr | `null` | Explicit gas-expulsion timescale. |
| `mdot_factor` | -- | `0.0` | Gas mass change rate relative to the stellar formation rate. |
| `eta_radius` | -- | `0.5` | Stellar Plummer scale radius relative to the initial gas scale radius. |
| `eta_sigma` | -- | `0.6` | Stellar velocity-dispersion scaling relative to the cloud virial velocity. |
| `kappa` | -- | `1.8` | Exponent controlling the radial velocity-dispersion profile. |
| `nworkers` | -- | `2` | PeTar worker count. |
| `t_end` | Myr | `30.0` | Final model time. |
| `dt_out` | Myr | `0.05` | Requested output interval, rounded to a power-of-two N-body time step. |
| `dt_level` | -- | `15` | PeTar softening-step parameter, setting `dt_soft = 2^-dt_level` in N-body units. |
| `stars_per_worker` | -- | `0` | Threshold for worker scaling. `0` disables scaling. |
| `track_background_gas_energy` | -- | `false` | Include the evolving background potential in energy diagnostics. |
| `test_background` | -- | `false` | Evaluate and plot the background-gas evolution instead of starting D-CAF. |
| `field_binaries` | -- | `false` | Apply the configured field-binary population to the initial stars. |
| `dry_run` | -- | `false` | Exit after parameter parsing. |

For example, a single command-line override can shorten a configuration run:

```shell
dcaf-plummer --config config.yaml --t_end 5.0
```

## Workflow

The runner performs the following steps:

1. Derive the cloud mass from `Mstars / sfe`, then determine the cloud
   free-fall time or radius.
2. Derive gas-evolution times and construct a `PlummerSphere` background
   potential.
3. Sample Kroupa IMF stellar masses and generate a Plummer realization.
4. Rescale the stellar positions and velocities using `eta_radius`,
   `eta_sigma`, and `kappa`.
5. Optionally replace the initial stars with a field-binary population.
6. Schedule formation through `MyFormationFramework`, a simple
   `StarFormationFramework` implementation supplied by the runner.
7. Configure PeTar, initialize `DcafSystem`, and evolve to `t_end`.

## Outputs

`DcafSystem` writes its main results in `./dcaf_output/`:

- `stars_###.amuse`: AMUSE stellar snapshots with saved `model_time` metadata.
- `energy.dat`: energy diagnostics at each output time.
- `dcaf.log`: runtime log. The runner sets this to debug level, so it includes
  timing diagnostics.

The Plummer background also writes `background_gas.dat` in the current working
directory. See the [output and snapshot API](../reference/dcaf/io/output.md)
and the [background-potential API](../reference/dcaf/backgroundgas/base.md) for
the file formats and loading helpers.

## Resume A Run

Resume uses the latest valid stellar snapshot and is supported only after star
formation has finished. It reads `config.yaml` by default, or a file explicitly
provided with `--config`:

```shell
dcaf-plummer --resume --config config.yaml --t_end 50.0
```

During resume, `t_end` is the only ordinary numerical command-line override.
The runner creates the next continuous output segment, such as
`dcaf_output_1/`, and retains the previous segment as the restore source. See
the [restart API](../reference/dcaf/io/restart.md) for the exact validation and
folder rules.

## Related APIs

- [DCAF system](../reference/dcaf/dcaf.md): solver lifecycle and output behavior.
- [Star-formation framework](../reference/dcaf/framework/starformation.md):
  formation scheduling and custom frameworks.
- [Plummer background potential](../reference/dcaf/backgroundgas/plummer.md):
  gas evolution model used by this runner.
- [Field binary population](../reference/dcaf/factory/field_binary_population.md):
  optional initial binary population.
