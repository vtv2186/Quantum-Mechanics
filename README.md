# Electron double slit

A CPU NumPy/SciPy solution of the **2D time-dependent Schrödinger equation**, not a prescribed interference texture. A Gaussian electron packet encounters a finite repulsive barrier with explicit apertures. A Strang split-step Fourier propagator evolves the complex wavefunction; smooth complex absorbing potentials suppress periodic FFT wraparound. The model is coherent, nonrelativistic and single-particle: no electron interactions, material lattice, spin, inelastic scattering or detector microscopic dynamics.

## Run

Python 3.12 is recommended. Install NumPy/SciPy/Matplotlib from `requirements.txt`; Manim additionally requires `ffmpeg pkg-config libcairo2-dev libpango1.0-dev` on Debian/Ubuntu. Use a virtual environment.

```bash
python -m pip install -r requirements.txt
python -m electron_slits --voltage 100 --separation-nm .982 --width-nm .246 --rate 100 --count 1000
python -m unittest discover -s tests -v
MPLCONFIGDIR=/tmp/matplotlib python -m manim -ql electron_slits/scene.py ElectronExperiment
```

`artifacts/detections.csv` contains seeded individual hits. `artifacts/wavefunction.npz` contains complex snapshots, the grid, time history, signed integrated screen current and conditional detection probabilities. `config.json` records simulation settings. Regenerate artifacts after changing physics; the scene reads them rather than rerunning propagation. `SLITS_ARTIFACTS` selects another artifact directory. Manim displays phase/amplitude and density evolution, then hits accumulating on **one fixed screen**. Its vertical hit jitter is display placement only; the measured coordinate is horizontal. Playback time is compressed independently of both physical clocks.

## Physics and units

For accelerating voltage U, central momentum is sqrt(2 m e U), wavelength h/p, length unit a=4 hbar/p, and time unit m a²/hbar. The internal TDSE is i dψ/dt=(-∇²/2+V-iW)ψ, with central k=4. The normalized packet starts at x=-22, with longitudinal standard deviation 3 and transverse standard deviation 20. Slit barrier: |x|<0.5, height 80 (ten times the central kinetic energy); aperture positions are ±d/2 and widths w. Closing one slit leaves the positive-y aperture open. The detector is x=60; edge absorption starts at x=-36, x=70 and |y|=50. Barrier tunneling is finite, rather than mathematically forbidden.

Default grid spacing 0.25, timestep 0.02, 1500 steps. Geometry except aperture width/separation scales with wavelength as voltage changes. Thus changing voltage also changes the physical screen distance and packet dimensions; this is explicitly a wavelength-scaled experiment. Arbitrary macroscopic apparatus dimensions are not feasible with this carrier-resolving grid. Apertures must span at least four cells and must fit inside the domain; refine `Config.dx` for narrow apertures. Increasing voltage can require refinement. Above 5 kV the implementation refuses the nonrelativistic approximation.

The detector integrates j_x=Im(ψ* ∂xψ) over physical propagation time. Integrated signed flux is saved; positive bins are normalized for conditional transmitted-electron sampling. A flux detector is appropriate for the outgoing packet with negligible recrossing/backflow, which tests check for defaults; it is not a general arrival-time POVM. Absorber loss is measured at each potential half-step. Remaining norm + cumulative absorption must equal one; **screen flux is diagnostic, not an additional removed probability**. The finite screen window and finite propagation interval limit collection. Detection times are an independent Poisson process at the requested **observed transmitted rate**, in seconds. They are not the femtosecond flight-time distribution. Blocked/reflected electrons are excluded from the CSV.

Expected paraxial fringe spacing is λL/d. Finite slit width gives a diffraction envelope; finite distance, finite momentum spread, grid resolution and finite barrier cause deviations. Closing one slit removes the two-aperture cross term but retains single-slit diffraction. Tests compare central maxima with λL/d, require a single central envelope peak for the closed slit, check Born sampling and separate clocks, conservation and timestep refinement. For quantitative studies vary dx, dt, domain, absorber and final time independently; default validation is not a universal error bound for arbitrary parameter choices.

## Databricks

Import `notebooks/DoubleSlit.py` as a Databricks source notebook. Set its repo-path widget to the checkout and output-path to a writable `/Volumes/<catalog>/<schema>/<volume>/...` for durable CSV/NPZ/MP4 files. The numerical workflow runs locally on the driver; it does not need Spark or a GPU. The notebook provides controls, animated complex phase and density, and the closed-slit comparison.

For optional rendering on a classic cluster, use an administrator-approved cluster init script installing `ffmpeg pkg-config libcairo2-dev libpango1.0-dev`, then `%pip install manim==0.19.2`. Serverless clusters may restrict native package installation: run the numerical cells there and render on a compatible classic cluster or local Python machine. No credentials or external services are required by the simulation.

## Validation in this environment

Seven numerical tests passed, including spatial and timestep refinement. Central fringe spacing: 7.375 internal units versus the de Broglie prediction 7.4934 (1.6% difference). Maximum probability budget error: 1.31e-14. Spectral-current screen crossing probability: 0.1150. See `artifacts/validation.txt` and `artifacts/preview.png`. The numerical workflow ran here; Databricks cluster execution was not available for verification. MP4 rendering was attempted but unavailable because Manim needs missing Cairo/Pango development headers and this environment has no system installation privileges. The scene is supplied, but no rendered MP4 is claimed.

## Open in VS Code

Clone this repository, open its folder in VS Code, and install the Microsoft Python extension. Create a virtual environment with `python -m venv .venv`, activate it (`source .venv/bin/activate` on macOS/Linux or `.venv\Scripts\Activate.ps1` in Windows PowerShell), and select that environment using **Python: Select Interpreter**. Install dependencies as described above; Manim's native dependencies are only needed for Manim rendering.

For the numerical simulation and Matplotlib previews alone:

```bash
python -m pip install numpy scipy matplotlib pillow
python -m electron_slits
python -c "from electron_slits import Config, run, save; save(run(Config(closed=True)), 'artifacts/single_slit')"
python -m electron_slits.visualize
```

Run these commands from the repository root. The visualization command reads both runs and writes `artifacts/evolution.gif`, `artifacts/detections.gif`, and `artifacts/comparison.png`. The committed `artifacts/evolution.mp4` is a preview converted from the GIF with FFmpeg; it is not a Manim render. Open it in a browser or video player if VS Code does not preview it.

To recreate that MP4 when FFmpeg is installed:

```bash
ffmpeg -y -i artifacts/evolution.gif -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p -movflags +faststart artifacts/evolution.mp4
```

Manim Voiceover has not yet been integrated.
