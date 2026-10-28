from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
r=np.load('artifacts/wavefunction.npz')
a=float(r['length_m'])*1e9
frames=r['snapshots']; vmax=np.max(abs(frames)**2)
extent=np.array([r['x'][0],r['x'][-1],r['y'][0],r['y'][-1]])*a
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
phase=axes[0].imshow(np.angle(frames[0].T),origin='lower',extent=extent,cmap='twilight',vmin=-np.pi,vmax=np.pi)
density=axes[1].imshow(abs(frames[0].T)**2,origin='lower',extent=extent,cmap='magma',vmin=0,vmax=vmax)
axes[0].set_title('Wavefunction phase / amplitude'); axes[1].set_title('Probability density')
for ax in axes:
    ax.set(xlabel='x (nm)',ylabel='y (nm)')
    ax.axvline(0,color='gray',lw=.7);ax.axvline(60*a,color='cyan',lw=.7)
clock=fig.suptitle('')
def evolution(i):
    f=frames[i].T;phase.set_data(np.angle(f));phase.set_alpha(np.sqrt(abs(f)**2/vmax));density.set_data(abs(f)**2)
    clock.set_text(f"Physical propagation time: {r['times'][i]*r['time_s']*1e15:.2f} fs")
FuncAnimation(fig,evolution,frames=len(frames),interval=200).save('artifacts/evolution.gif',writer=PillowWriter(fps=5),dpi=90)
plt.close(fig)
hits=np.genfromtxt('artifacts/detections.csv',delimiter=',',names=True)
fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
ax.set(xlim=(-64*a,64*a),ylim=(-1.5,1.5),xlabel='Measured transverse screen position (nm)',yticks=[],title='Electron hits on one fixed screen')
ax.set_facecolor('#091527'); dots=ax.scatter([],[],s=5,c='#ffd34f',alpha=.8)
label=ax.text(.02,.95,'',transform=ax.transAxes,color='white',va='top')
rng=np.random.default_rng(42); jitter=rng.uniform(-1.35,1.15,len(hits))
arrivals=hits['observation_time_s']; positions=hits['screen_y_m']*1e9
end=arrivals[-1]
def accumulate(i):
    t=end*i/49;mask=arrivals<=t
    dots.set_offsets(np.column_stack([positions[mask],jitter[mask]]))
    label.set_text(f'Observation time: {t:.2f} s | Hits: {mask.sum()}')
FuncAnimation(fig,accumulate,frames=50,interval=100).save('artifacts/detections.gif',writer=PillowWriter(fps=10),dpi=90)
plt.close(fig)
s=np.load('artifacts/single_slit/wavefunction.npz')
fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
ax.plot(r['y']*a,r['probability']/((r['y'][1]-r['y'][0])*a),label='Both slits open')
ax.plot(s['y']*a,s['probability']/((s['y'][1]-s['y'][0])*a),label='One slit closed')
ax.set(xlabel='Screen position (nm)',ylabel='Conditional density (1/nm)',title='Simulated screen probabilities');ax.legend()
fig.savefig('artifacts/comparison.png',dpi=140);plt.close(fig)
print('Created evolution.gif, detections.gif, comparison.png')
