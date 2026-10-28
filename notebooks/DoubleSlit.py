# Databricks notebook source
# MAGIC %md
# MAGIC # Electron double slit: TDSE and Born detections
# MAGIC Run in VS Code/Jupyter or on a Databricks CPU cluster. Locally, edit the `parameters` dictionary below. On Databricks, use the widgets and optionally select a writable Unity Catalog Volume for outputs. Manim requires native Cairo/Pango headers; see README. No Spark execution is needed.

# COMMAND ----------
# MAGIC %pip install "numpy>=2.0,<3" "scipy>=1.13,<2" "matplotlib>=3.8,<4" pillow

# COMMAND ----------
import os, sys
from pathlib import Path
# VS Code/Jupyter: edit these values, then rerun this cell.
# Discover the checkout whether the kernel starts in its root or notebooks folder.
def find_repo():
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / 'electron_slits' / '__init__.py').is_file():
            return candidate
    raise FileNotFoundError('Open this notebook from the cloned Quantum-Mechanics folder, or set repo_path explicitly.')

parameters = {
    'repo_path': str(find_repo()),
    'output_path': str(find_repo() / 'artifacts' / 'notebook'),
    'voltage_V': '100',
    'separation_nm': '0.982',
    'width_nm': '0.246',
    'observed_rate_Hz': '100',
}
if 'dbutils' in globals():
    for name, default in parameters.items():
        dbutils.widgets.text(name, default)
    parameters = {name: dbutils.widgets.get(name) for name in parameters}
repo = Path(parameters['repo_path']).expanduser().resolve()
if not (repo / 'electron_slits' / '__init__.py').is_file():
    raise FileNotFoundError(f'repo_path must contain electron_slits: {repo}')
sys.path.insert(0, str(repo))
from electron_slits import Config, run, save
config = Config(voltage=float(parameters['voltage_V']),
                separation_nm=float(parameters['separation_nm']),
                width_nm=float(parameters['width_nm']))
result = run(config)
out = Path(parameters['output_path']).expanduser().resolve()
save(result, out, rate=float(parameters['observed_rate_Hz']))
print('Screen crossing probability:', result['transmission'])
print('Probability budget error:', abs(result['history'][:,3]-1).max())

# COMMAND ----------
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
frames = result['snapshots']
extent = np.array([result['x'][0], result['x'][-1], result['y'][0], result['y'][-1]]) * result['length_m'] * 1e9
fig, axes = plt.subplots(1,2,figsize=(12,5))
im0=axes[0].imshow(np.angle(frames[0].T),origin='lower',extent=extent,cmap='twilight',vmin=-np.pi,vmax=np.pi,alpha=np.sqrt(abs(frames[0].T)**2/np.max(abs(frames)**2)))
im1=axes[1].imshow(abs(frames[0].T)**2,origin='lower',extent=extent,vmin=0,vmax=np.max(abs(frames)**2))
for ax in axes:
    ax.set(xlabel='Propagation x (nm)',ylabel='Transverse y (nm)')
    ax.axvline(0,color='white',alpha=.4); ax.axvline(60*result['length_m']*1e9,color='red',alpha=.4)
axes[0].set_title('Phase (brightness: amplitude)'); axes[1].set_title('Probability density')
def update(i):
    f=frames[i].T; im0.set_data(np.angle(f)); im0.set_alpha(np.sqrt(abs(f)**2/np.max(abs(frames)**2))); im1.set_data(abs(f)**2)
    fig.suptitle(f"Physical flight time {result['times'][i]*result['time_s']*1e15:.3f} fs")
    return im0,im1
ani=FuncAnimation(fig,update,frames=len(frames),interval=120)
display(HTML(ani.to_jshtml())); plt.close(fig)

# COMMAND ----------
fig,ax=plt.subplots()
ax.plot(result['y']*result['length_m']*1e9,result['probability'],label='Double slit')
single=run(Config(**{**result['config'],'closed':True}))
ax.plot(single['y']*single['length_m']*1e9,single['probability'],label='One slit closed')
ax.set(xlabel='Screen y (nm)',ylabel='Conditional probability per bin'); ax.legend(); display(fig); plt.close(fig)
# CSV has observation_time_s and screen_y_m. Observation time is independent of flight time.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Optional MP4 render
# MAGIC After native dependencies and `manim==0.19.2` are installed, run the next cell. Use a Volume for persistence. The video is embedded without a localhost server.

# COMMAND ----------
import subprocess, base64
# Set True after installing the optional render dependencies.
render_video = False
if render_video:
    env={**os.environ,'SLITS_ARTIFACTS':str(out),'MPLCONFIGDIR':'/tmp/matplotlib'}
    subprocess.run([sys.executable,'-m','manim','-ql',str(repo/'electron_slits/scene.py'),'ElectronExperiment','--media_dir',str(out/'media')],cwd=repo,env=env,check=True)
    video=next((out/'media').rglob('ElectronExperiment.mp4'))
    display(HTML('<video controls src="data:video/mp4;base64,'+base64.b64encode(video.read_bytes()).decode()+'"></video>'))
