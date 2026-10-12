# Instructions for Claude

## GPU time window

Do not run anything that uses the GPU between 08:00 and 18:00 Paris time. This covers policy
rollouts, grid runs, smoke tests, timing runs and any script that loads a model onto the GPU.

- Check the time first with `TZ=Europe/Paris date`.
- Inside the window, prepare the command or script and tell me it is ready to launch after
  18:00. Do not start it, and do not schedule it to start later without asking.
- A run that would still be going at 08:00 must not be started unless I say so.
- CPU-only work (statistics, tests on simulated data, plots, config and code edits) is fine at
  any time.
