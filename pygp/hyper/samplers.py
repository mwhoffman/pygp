"""
This package implements slice sampling and is taken more-or-less directly from
the Spearmint package of Snoek et al.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np

# exported symbols
__all__ = ['slice_sample']


def slice_sample(logprob, x0, sigma=1.0, step_out=True, max_steps_out=1000):
    def direction_slice(direction, x0):
        def dir_logprob(z):
            return logprob(direction*z + x0)

        upper = sigma*np.random.rand()
        lower = upper - sigma
        llh_s = np.log(np.random.rand()) + dir_logprob(0.0)

        l_steps_out = 0
        u_steps_out = 0
        if step_out:
            while dir_logprob(lower) > llh_s and l_steps_out < max_steps_out:
                l_steps_out += 1
                lower -= sigma
            while dir_logprob(upper) > llh_s and u_steps_out < max_steps_out:
                u_steps_out += 1
                upper += sigma

        while True:
            new_z = (upper - lower)*np.random.rand() + lower
            new_llh = dir_logprob(new_z)
            if np.isnan(new_llh):
                raise Exception("Slice sampler got a NaN")
            if new_llh > llh_s:
                break
            elif new_z < 0:
                lower = new_z
            elif new_z > 0:
                upper = new_z
            else:
                raise Exception("Slice sampler shrank to zero!")

        return new_z*direction + x0

    # FIXME: I've removed how blocks work because I want to rewrite that bit. so
    # right now this samples everything as one big block.
    direction = np.random.randn(x0.shape[0])
    direction = direction / np.sqrt(np.sum(direction**2))
    return direction_slice(direction, x0)
