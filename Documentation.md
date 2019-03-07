## Parallelization intuition

The Gale and Shapley algorithm for finding a stable matching in an instance follows a proposal/deferred acceptance mechanism.

One of the interesting properties of the algorithm is that, the final output is independent of the order in which the proposals occur. This is where we believe parallelization could be exploited.

However in the worst case scenario, if all the proposals are directed to the same person at a given time step, it could lead to inherent serialization, which we can tackle with atomic operations.

Hence we plan to focus on observing the speedup on an average case scenario by taking a mean over multiple instances.