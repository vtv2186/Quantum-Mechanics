import argparse
from .simulation import Config, run, save
p=argparse.ArgumentParser()
p.add_argument('--voltage',type=float,default=100)
p.add_argument('--separation-nm',type=float,default=.982)
p.add_argument('--width-nm',type=float,default=.246)
p.add_argument('--rate',type=float,default=100)
p.add_argument('--count',type=int,default=1000)
p.add_argument('--closed',action='store_true')
a=p.parse_args()
r=run(Config(voltage=a.voltage,separation_nm=a.separation_nm,width_nm=a.width_nm,closed=a.closed))
save(r,count=a.count,rate=a.rate)
print('Screen crossing probability:',r['transmission'])
print('Maximum probability budget error:',abs(r['history'][:,3]-1).max())
