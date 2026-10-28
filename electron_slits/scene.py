"""Run from repo root: manim -ql electron_slits/scene.py ElectronExperiment."""
import os
from pathlib import Path
import numpy as np
from manim import *

class ElectronExperiment(Scene):
    def construct(self):
        root=Path(os.environ.get('SLITS_ARTIFACTS','artifacts'))
        data=np.load(root/'wavefunction.npz')
        hits=np.genfromtxt(root/'detections.csv',delimiter=',',names=True)
        title=Text('Electron double slit',font_size=30).to_edge(UP)
        self.add(title)
        # Color encodes phase; brightness encodes amplitude. Density is shown separately.
        frames=data['snapshots']; scale=max(np.max(abs(f)**2) for f in frames)
        from matplotlib.colors import hsv_to_rgb
        images=[]
        for f in frames:
            density=abs(f.T)**2
            hsv=np.stack(((np.angle(f.T)+np.pi)/(2*np.pi),np.ones_like(density),np.sqrt(density/scale)),axis=-1)
            rgb=(255*hsv_to_rgb(hsv)).astype(np.uint8)
            gray=(255*np.sqrt(density/scale)).astype(np.uint8)
            images.append((rgb,np.repeat(gray[:,:,None],3,axis=2)))
        phase=ImageMobject(images[0][0]).set(width=6).move_to(LEFT*3.3)
        density=ImageMobject(images[0][1]).set(width=6).move_to(RIGHT*3.3)
        labels=VGroup(Text('Phase / amplitude',font_size=20).move_to([-3.3,2.5,0]),Text('Probability density',font_size=20).move_to([3.3,2.5,0]))
        clock=Text('',font_size=18)
        self.add(phase,density,labels)
        for i,(rgb,gray) in enumerate(images):
            phase.become(ImageMobject(rgb).set(width=6).move_to(LEFT*3.3))
            density.become(ImageMobject(gray).set(width=6).move_to(RIGHT*3.3))
            self.remove(clock)
            clock=Text(f"Physical propagation: {data['times'][i]*data['time_s']*1e15:.2f} fs",font_size=20).to_edge(DOWN)
            self.add(clock); self.wait(.10)
        self.remove(phase,density,labels,clock)
        screen=Rectangle(width=11,height=3,color=BLUE)
        self.add(screen,Text('Fixed detector screen (transverse position)',font_size=22).move_to([0,2,0]))
        t=ValueTracker(0); end=float(hits['observation_time_s'][-1])
        counter=always_redraw(lambda: Text(f'Observation: {t.get_value():.2f} s',font_size=22).to_edge(DOWN))
        self.add(counter)
        # Vertical jitter represents display placement only, not another measured coordinate.
        rng=np.random.default_rng(42)
        extent=max(abs(data['y']))*float(data['length_m'])
        dots=VGroup(*[Dot([float(y)/extent*5.3,float(z),0],radius=.023,color=YELLOW).set_opacity(0)
                     for y,z in zip(hits['screen_y_m'],rng.uniform(-1.35,1.35,len(hits)))])
        arrivals=hits['observation_time_s']
        def reveal(group):
            for dot,arrival in zip(group,arrivals): dot.set_opacity(float(arrival<=t.get_value()))
        dots.add_updater(reveal); self.add(dots)
        self.play(t.animate.set_value(end),run_time=5,rate_func=linear)
        dots.clear_updaters(); self.wait(.5)
