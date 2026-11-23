# Benchmarks you can rerun

This folder is for comparisons a reader can rerun and check, not for marketing numbers. Each harness here follows
the same rules:

1. **Same budget for every solver.** The same wall time (or the same time to a stated target), on hardware that is
   named. When the 4colors service runs on its own machines and a local solver runs on yours, the README says so.
2. **Same problem, same answer check.** Every answer is decoded and its objective recomputed in plain Python by one
   scorer, for every solver. A solver's own reported objective is kept beside it, never used to rank.
3. **Seeds and repeats.** Several seeds per cell; the README states how many, and reports the median and the spread.
4. **Public instances only**, loaded by `datasets/` loaders that state each set's license and source.
5. **Versions recorded.** Every row records the `fourcolors` client version, the engine version the service
   reports (`engine_version`), and each local solver's version.
6. **Only solvers whose license allows published comparisons**, run as their documentation recommends.

Planned harnesses: QUBO (open-source samplers you can `pip install`), and compiler (Qiskit's own `transpile` at
the same wall time). Each will ship with tests and a `--dry-run` that prints the plan and the job budget
before anything is billed.
