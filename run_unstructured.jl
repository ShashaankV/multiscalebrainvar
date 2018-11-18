#include("random_2x2_fast.jl")
#using SpecialFunctions
#requires Pkg.add("Random")
using Distributions
include("2x2_measure_inputs.jl")
#println("This code simulates a random EI network.
#Each population receives an independent but statistically identical copy of an OU process as input.
#Inhibitory neurons receive slightly weaker input than excitatory neurons.
#This code shows that the network is tracking differences in the inputs.")

include("analyze.jl")
srand(4321)

#Coupling parameters
Aee = 12.5
Aie = 20
Aei = 50.
Aii = 50.

#Population size paramters
N = 4000
IFRAC = 2.
Ni = Int64(round(N/IFRAC))
Ne = N - Ni
k = 600
ks = sqrt(k)
k2 = round(Int64, k/2)
Ne2 = round(Int64, Ne/2)
Ni2 = round(Int64, Ni/2)

#Time
runtime = 10000 #ms
h = .1 #step size
ntotal = round(runtime/h) #time points
rt = runtime/1000 #runtime in seconds
rot = Int64(ntotal)
rot2 = Int64(ntotal*2)

#parameters for measuring spiking statistics
min_e_neurons = 20
min_i_neurons = 50
min_spikes_e = 10
min_spikes_i = 10
fbinsize = 400/h
cbinsize = 100/h
netd_binsize = 25/h

#Drive parameters: fe1=fe2, fi1=fi2
stdev = 1.
tau_n = 500
mu = .2

#Two independent random gaussian variables with 0 mean
R1 = rand(Normal(0., stdev), rot2);
R2 = rand(Normal(0., stdev), rot2);

#Pass them through a first order ODE: dx/dt = -x/tau + noise
s1 = OU_Model(R1, tau_n, h);
s2 = OU_Model(R2, tau_n, h);

#Take the resulting time-series and add a bias
s1 .+= mu;
s2 .+= mu;

#Use the second half of the time series, after burn-in
fe1 = s1[rot:end];
fe2 = s2[rot:end];

#Drive to inhibitory neurons has a weaker bias than excitatory
fi1 = fe1 .- (0.1*h);
fi2 = fe2 .- (0.1*h);

#Neuron parameters
vth = 20
tau_m = 20.
tau_s = 2.

#Generate weights matrix and sparse representation
W = local_random_2x2_symmetric(N, IFRAC, k, Aee, Aei, Aie, Aii);
CSR = sparse_rep(W, N);

#Do a simulation
@time t, r = euler_lif_2x2_CSR_OU(h, runtime, N, IFRAC, W, CSR, fe1, fi1, fe2, fi2, vth, tau_m, tau_s)

#Chunk raster into excitatory and inhibitory components
e_m = find(r .<= Ne);
i_m = find(r .> Ne);
te = t[e_m];
re = r[e_m];
ti = t[i_m];
ri = r[i_m];

#Dominance time metric
#D(t) = (A-B)/(A+B); ntd is numeratory (the difference in activity levels), nts is denominator (sum of activity levels)
ntd, nts = nt_diff_H(te, re, rot, Ne2, netd_binsize);
s = ntd ./ nts;

#Dominance metric for drive terms requires computing difference / sum
sd = fe2 .- fe1;
ss = fe2 .+ fe1;
sA = sd ./ ss;

#Downsample input time series to be the same size as s
sA_S = downsample(sA, s);

#standardize time series
sA_S2z = zscore(sA_S);
s2z = zscore(s);
#plot(sA_S2z);
#plot(s2z);

#write results to a file for later use
write_raster("rast.txt", t, r)
write_array("inputz.txt", sA_S2z)
write_array("ratez.txt", s2z)
write_array("input1.txt", fe1)
write_array("input2.txt", fe2)
