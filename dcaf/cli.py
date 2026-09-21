"""Console-command entry points for D-CAF workflows."""


def parametric_plummer() -> None:
    """Run the parametric Plummer simulation command."""
    from dcaf.scripts.parametric_plummer import main, parse_args, save_params

    params = parse_args()
    if not params.get("resume", False):
        save_params(params)
    main(**params)
