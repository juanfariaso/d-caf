# D-CAF – Dynamic Cluster Assembly Framework

**D-CAF** is an AMUSE-based framework for producing *N-body* simulations of star cluster assembly under prescribed star formation histories.
Gas is not explicitly modelled but emulated through ad-hoc evolving background potentials.

The framework aims to mimic full star cluster formation simulations while making explicit assumptions about gas behaviour and star formation.
This enables exploration of the consequences of such assumptions on stellar dynamics, stellar populations, and the long-term evolution of clusters.


# Documentation

See the [documentation webpage](https://juanfariaso.github.io/d-caf).

# Install
see the [installation notes](https://juanfariaso.github.io/d-caf/install/).

# Code organization

Here how the repository is currently organized. Future changes should respect this structure for simplicity:

---
dcaf/dcaf.py — the orchestrator

Owns the simulation lifecycle and wiring:
	•	initialisation (codes, units, converters)
	•	main evolve loop + output cadence
	•	Bridge coupling and timestep bookkeeping
	•	energy / diagnostics bookkeeping that must be done during the run
	•	calling the framework to form stars
	•	applying the background gas potential and its evolution

Here goes any feature that affects the runtime loop, restart/output policy, worker scaling, Bridge/Petar wiring, or global bookkeeping.

---

dcaf/framework/ — star formation rules

Defines when/where/how stars are formed:
	•	formation scheduling (e.g. next_formation_time())
	•	star creation (e.g. form_stars(...))
	•	connections to IC generators (factory samplers)
	•	SFR / target-star logic

Here are the features that changes the star-formation model, IMF sampling, cluster assembly logic, or anything that conceptually belongs to the star formation framework

---

dcaf/factory/ — initial-condition generators / samplers

Reusable generators for spatial/kinematic structure:
	•	fractal / substructure recipes
	•	distance-based placement / other geometric samplers
	•	any “give me positions/velocities/masses for N stars” helper

Here goes sampling and initial condition methods that can be used by the different frameworks. Ideally these are universal methods that can be reused. 

---

dcaf/backgroundgas/ — background potential models

Background gas potentials and their evolution:
	•	base class + shared interface in base.py
	•	concrete potentials (e.g. plummer.py)
	•	time-dependent mass/scale evolution
	•	computing Φ(r), a(r), etc.

⸻

dcaf/utilities/ — shared plumbing

Common helpers that should not depend on a specific framework/potential:
	•	config.py: YAML loading, validation, normalisation
	•	parameters.py: defaults and parameter validation helpers
	•	logger.py: logging setup
	•	small generic helpers/samplers

Here goes any config knobs, validation rules, logging conventions, or generic utilities reused across modules.

---

dcaf/analysis/ — post-processing

Offline analysis on saved outputs:
	•	formation histories
	•	multiplicity analysis
	•	derived statistics from snapshots

Here goes the methods that can analyze the output of d-caf.

---

dcaf/tests/ — invariants and regression tests

Small tests to validate any feature.
The goal here is that any new feature should come with at least one test that would fail without it.
(I am have not been very rigurouse here.. TODO)

⸻

examples/ — runnable reference scripts.

TODO: Examples are not updated to the latest version.

Minimal scripts demonstrating typical configurations.
Should be the landing place for new users.

⸻

scripts/ — general standalone scripts
	•	analysis scripts
	•	batch helpers
	•	convenience runners
