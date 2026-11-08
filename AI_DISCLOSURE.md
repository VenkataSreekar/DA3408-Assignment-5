# AI Disclosure

**Name:** Venkata Sreekar · **Roll no:** DA24B030 · **Assignment:** Module 4, Assignment 5 (SwiftParcel ETA service)

## Tool used
Claude (Anthropic), via the claude.ai chat interface, with its code-execution sandbox.

## What the AI did
- **Code:** drafted the first versions of `partA/eta_app.py`, `test_eta_app.py`, `run_a3.py`; `partB/eta_mlflow.py`, `test_gate.py`, `shadow_compare.py`; `partC/eta_ray.py`, `verify_ray.py`; and `partD/smoke_eta.py`. It also suggested the helper files `demo_ray.py` and `ask.sh` used in the demo video.
- **Documents:** drafted `README.md` and `REPORT.md`.
- **Debugging:** helped diagnose problems I ran into (for example, the `ask` shell function error and the registry alias display).

## What I did
- Set up the environment (conda env `aiops`, Python 3.13) and ran everything on my own machine: tests, load tests, MLflow registration, gate, shadow comparison, Ray verification and the smoke test.
- All measurements and outputs in `REPORT.md`, `gate_output.txt`, `shadow_output.txt`, `verify_output.txt`, `loadtest_results*.json` come from my own runs. Numbers the AI produced in its own sandbox (a different, 1-CPU machine) were not used.
- Machine description, power conditions and the interpretation of my results.
- Recorded the demo video and created and pushed the repository.

## Sources besides the AI
- Course repository and Module 4 materials: https://github.com/sudarsun/DA3408. Ideas reused from the lecture notebooks: TODO (name each notebook and what I took, or "none").
- Starter files `eta_data.py`, `load_test.py`, `requirements.txt`, used unmodified.
- Official documentation for FastAPI/pydantic, MLflow, Ray Serve and the Prometheus text format.


## Policy
I have followed the course's AI fair-use code of conduct.