import numpy as np


def get_key_positions(particles, keys):
    """
    Return the positions of `keys` in `particles`, preserving the input order.

    This function is subset-safe: returned positions are valid for indexing the
    visible `particles` set directly.

    It fails fast if:
    - any requested key is missing
    - the particle set contains duplicated keys
    - the requested key list contains duplicated keys
    """
    particle_keys = np.asarray(particles.key)
    keys = np.asarray(keys)

    assert len(np.unique(particle_keys)) == len(particle_keys), (
        "[DCAF][KEYS] Particle set contains duplicated keys."
    )
    assert len(np.unique(keys)) == len(keys), (
        "[DCAF][KEYS] Requested key list contains duplicated keys."
    )

    order = np.argsort(particle_keys)
    sorted_keys = particle_keys[order]
    positions = np.searchsorted(sorted_keys, keys)

    assert not np.any(positions >= len(sorted_keys)), (
        "[DCAF][KEYS] Some requested keys are missing from the particle set."
    )

    matched_keys = sorted_keys[positions]
    assert np.array_equal(matched_keys, keys), (
        "[DCAF][KEYS] Some requested keys are missing from the particle set."
    )

    return order[positions]
