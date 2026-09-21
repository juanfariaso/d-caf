# Static Plummer Background

This tutorial builds a D-CAF model directly in Python. It demonstrates the two
main extension points: a custom
[StarFormationFramework](../reference/dcaf/framework/starformation.md) and a
custom [background potential](../reference/dcaf/backgroundgas/base.md). These
base classes are intended to be inherited by user-defined classes. The complete
example is
[`examples/static_plummer_example.py`](https://github.com/juanfariaso/d-caf/blob/main/examples/static_plummer_example.py).

The model contains a Plummer stellar population embedded in a static Plummer
gas potential. Stars appear at a constant mass rate throughout the embedded
phase. At its end, the example removes the gas potential instantaneously.

## Run It

From the repository root, with AMUSE and PeTar installed:

```shell
python examples/static_plummer_example.py
```

The run writes stellar snapshots and diagnostics to `tutorial_output/`.

## 1. Define The Physical Scale

The chosen star-formation efficiency gives the initial cloud mass,

$$
M_{\rm cloud} = \frac{M_\star}{\epsilon}.
$$

The example uses `300 MSun` of stars, `sfe = 0.3`, a one-Myr embedded phase,
and a one-parsec gas Plummer radius. Change these values at the top of `main()`
to define a different model.

## 2. Create Stars And Field Binaries

`make_stars()` draws Kroupa masses, rescales their sum to the requested stellar
mass, creates an AMUSE Plummer realization, and applies
[FieldBinaryPopulation](../reference/dcaf/factory/field_binary_population.md):

```python
binaries = FieldBinaryPopulation(
    mult_frac="field",
    pdist="inner",
    qdist="field",
    edist="field",
    min_mass=0.08,
)
stars = new_plummer_model(nstars, convert_nbody=converter)
stars.mass = masses
stars = binaries.apply(stars)["resolved_stars"]
```

The returned resolved particles include binary companions and the bookkeeping
needed for D-CAF to keep each system together during formation scheduling. See
the [field-binary API](../reference/dcaf/factory/field_binary_population.md)
for the available population prescriptions. The example uses a `0.1 MSun` IMF
floor so each selected primary can receive a companion above the field model's
minimum companion mass.

## 3. Write A Formation Framework

The custom framework is intentionally minimal:

```python
class ConstantSFR(StarFormationFramework):
    def form_stars(self, active_stars):
        return self.extract_next_event()
```

[StarFormationFramework](../reference/dcaf/framework/starformation.md)
constructs the constant-SFR schedule from the target particles. The override
must call `extract_next_event()` exactly once. More complex user-defined
subclasses can use `active_stars` here to choose new positions, velocities, or
masses at each event.

The example chooses the rate so that the resolved stellar mass forms during the
embedded phase:

$$
\dot{M}_\star = \frac{M_{\star,\rm resolved}}{t_{\rm embedded}}.
$$

## 4. Write A Background-Gas Model

The gas model subclasses
[PlummerSphere](../reference/dcaf/backgroundgas/plummer.md), which already
supplies the Bridge gravity interface. Its only customization is the
post-embedded evolution:

```python
class InstantaneousExpulsionPlummer(PlummerSphere):
    def evolve_model(self, tend):
        if tend < self.t_ge:
            super().evolve_model(tend)
            return
        super().evolve_model(self.t_ge)
        self.mtot = 0 | self.mtot.unit
        self.model_time = tend
```

This leaves the gas mass and radius constant before `t_ge`, then makes the
background force vanish afterward. For gradual gas evolution, inherit from
`PlummerSphere` and replace this evolution rule.

## 5. Initialize And Evolve D-CAF

Load the default D-CAF configuration, then set the PeTar time-step level. The
softening time step must be an exact power of two in N-body time units; using
this block-step form is also efficient for PeTar:

```python
config = get_default_configuration()
config["petar"].number_of_workers = 1
dt_level = 15
config["petar"].dt_soft = 2**-dt_level | nbody_system.time
```

The final block supplies the framework, AMUSE converter, and configuration to
[DcafSystem](../reference/dcaf/dcaf.md), then selects the output interval and
final time:

```python
system = DcafSystem(
    framework=framework,
    config=config,
    converter=converter,
    output_folder="tutorial_output",
)
system.dt_out = 0.1 | units.Myr
system.initialize_system()
system.evolve_model(end_time)
```

From here, substitute your own formation framework or gas subclass while
preserving the same `DcafSystem` lifecycle.
