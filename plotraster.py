from matplotlib.pyplot import *
ion()
from numpy import *
h=.1
f=open(sys.argv[1],'r')
a=f.read()
f.close()
a=a.split('\n')[:-1]
t=[]
n=[]
t0=0/h #use to plot chunk of time
t1=10000/h
nind0=0 #use to plot chunk of neurons
nind1=2000#<2k are excitatory for unstructured and discrete; <3.2k for continuum
for i in a:
    i=i.split(',')
    if float(i[0])>t0 and float(i[0])<t1 and float(i[1])>nind0 and float(i[1])<nind1:
        t.append(float(i[0]))
        n.append(float(i[1]))
    if float(i[0])>t1:
        break

n=array(n)
t=array(t)*h #convert to milliseconds

scatter(t,n,1,'k','s',alpha=.1)


input()
