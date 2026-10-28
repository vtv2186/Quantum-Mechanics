import unittest
import numpy as np
from scipy.signal import find_peaks
from electron_slits import Config, run, detections

class PhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.double=run()
        cls.single=run(Config(closed=True))
    def test_probability_budget(self):
        for r in (self.double,self.single):
            self.assertLess(np.max(abs(r['history'][:,3]-1)),1e-10)
            self.assertTrue(np.all(np.diff(r['history'][:,2])>=-1e-12))
            self.assertGreater(r['transmission'],.005)
            self.assertLess(r['transmission'],1.)
            self.assertLess(abs(np.minimum(r['flux'],0).sum())/(abs(r['flux']).sum()),.01)
    def test_unitary_without_absorber(self):
        r=run(Config(barrier=False,absorb=False,steps=100),snapshot_every=20)
        self.assertLess(np.max(abs(r['history'][:,1]-1)),1e-11)
        self.assertEqual(r['history'][-1,2],0)
    def test_fringe_spacing_and_closed_slit(self):
        r=self.double; y=r['y']; p=r['probability']
        peaks,_=find_peaks(p,prominence=p.max()*.1,distance=20)
        near=y[peaks][abs(y[peaks])<20]
        measured=np.median(np.diff(near))
        d=Config().separation_nm*1e-9/r['length_m']
        predicted=(2*np.pi/4)*60/d
        print('Fringe spacing internal units:',measured,'de Broglie:',predicted)
        self.assertGreaterEqual(len(near),3)
        self.assertLess(abs(measured/predicted-1),.20)
        # Single slit retains diffraction, but loses the narrow two-aperture fringes.
        peaks1,_=find_peaks(self.single['probability'],prominence=self.single['probability'].max()*.15,distance=20)
        self.assertEqual(np.count_nonzero(abs(y[peaks1])<20),1)
    def test_born_sampling_and_clocks(self):
        t,y=detections(self.double,20000,rate=100,seed=10)
        self.assertTrue(np.all(np.diff(t)>0))
        self.assertLess(abs(t[-1]/200-1),.03)
        expected=np.sum(self.double['y']*self.double['probability'])*self.double['length_m']
        self.assertLess(abs(y.mean()-expected),.05*y.std())
        t2,y2=detections(self.double,20000,rate=200,seed=10)
        np.testing.assert_allclose(t2,t/2); np.testing.assert_array_equal(y,y2)
    def test_timestep_refinement(self):
        fine=run(Config(dt=.01,steps=3000),snapshot_every=3000)
        self.assertLess(np.sum(abs(fine['probability']-self.double['probability'])),.08)
    def test_spatial_refinement(self):
        fine=run(Config(dx=.2),snapshot_every=1500)
        q=np.interp(self.double['y'],fine['y'],fine['probability'])
        q/=q.sum()
        self.assertLess(np.sum(abs(q-self.double['probability'])),.15)
    def test_invalid_parameters(self):
        with self.assertRaises(ValueError): run(Config(voltage=-1))
        with self.assertRaises(ValueError): detections(self.double,rate=0)

if __name__=='__main__': unittest.main()
