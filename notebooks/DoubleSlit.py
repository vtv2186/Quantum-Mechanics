# Databricks notebook source
# MAGIC %md
# MAGIC # Electron double slit: TDSE and Born detections
# MAGIC Run on a CPU cluster. Set `repo_path` to this checkout and `output_path` to a writable Unity Catalog Volume for durable artifacts. Manim requires native Cairo/Pango headers; see README for a cluster init script. No Spark execution is needed for the FFT solver.

# COMMAND ----------
# MAGIC %pip install numpy>=2.0 scipy>=1.13 matplotlib>=3.8

# COMMAND ----------
import os, sys
from pathlib import Path
dbutils.widgets.text('repo_path', os.getcwd())
dbutils.widgets.text('output_path', '/tmp/electron-slits')
dbutils.widgets.text('voltage_V', '100')
dbutils.widgets.text('separation_nm', '0.982')
dbutils.widgets.text('width_nm', '0.246')
dbutils.widgets.text('observed_rate_Hz', '100')
repo = Path(dbutils.widgets.get('repo_path'))
sys.path.insert(0, str(repo))
from electron_slits import Config, run, save
config = Config(voltage=float(dbutils.widgets.get('voltage_V')),
                separation_nm=float(dbutils.widgets.get('separation_nm')),
                width_nm=float(dbutils.widgets.get('width_nm')))
result = run(config)
out = Path(dbutils.widgets.get('output_path'))
save(result, out, rate=float(dbutils.widgets.get('observed_rate_Hz')))
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
# MAGIC After native dependencies and `manim==0.19.2` are installed, run the next cell. Use a Volume for persistence. `displayHTML` embeds the video without a localhost server.

# COMMAND ----------
import subprocess, base64
# Set True after installing the optional render dependencies.
render_video = False
if render_video:
    env={**os.environ,'SLITS_ARTIFACTS':str(out),'MPLCONFIGDIR':'/tmp/matplotlib'}
    subprocess.run([sys.executable,'-m','manim','-ql',str(repo/'electron_slits/scene.py'),'ElectronExperiment','--media_dir',str(out/'media')],cwd=repo,env=env,check=True)
    video=next((out/'media').rglob('ElectronExperiment.mp4'))
    displayHTML('<video controls src="data:video/mp4;base64,'+base64.b64encode(video.read_bytes()).decode()+'"></video>')
