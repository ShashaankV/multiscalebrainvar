include("continuum.jl")
include("analyze.jl")

println("This code simulates rivalry in the continuum network with adaptation.
Same parameters used for Figure 3 and 4.
This code is specifically designed to check how skewness changes as a function of drive strength for reviewer #1")

using SpecialFunctions
using Random
using Statistics
using Distributions
Random.seed!(2134)

#Coupling widths
kee = .26
kei = .93
kie = .97
kii = .5
#Coupling strengths
Aee = 84.
Aei = 314.
Aie = 1319.
Aii = 689.
#Convert coupling widths from units of standard deviations to \kappa
kee = sd_2_k(kee)
kei = sd_2_k(kei)
kie = sd_2_k(kie)
kii = sd_2_k(kii)
#connection density
p = .34
#Population sizes
N = 4000
Ne = div(N*4, 5)
Ni = div(N, 5)
#Neuron parameters
vth = 20
tau_m = 20.
tau_s = 2.
tau_a = 650.
g_a = 0.013 #adaptation strength
#Feedforward input to excitatory neurons

#Time
runtime = 250*1000#ms
h = .1 #time step
ntotal = round(runtime/h) #time points
rt = ((ntotal)/1000.)*h #runtime in seconds
Ne2 = div(Ne, 2)
Ni2 = div(Ni, 2)
#A_ij = \int(0, 2pi) W_ij, scaling by p makes this density-invariant
Aee /= p
Aei /= p
Aie /= p
Aii /= p

#Inputs are box-shaped on the ring, in orthogonal locations, width parameter not used in these sims
input_width = 5.
ang = 0
#Generate weights matrix and sparse representation
W = Weights(Ne,Ni,kee,kei,kie,kii,Aee,Aei,Aie,Aii,p)
CSR = sparse_rep(W, N)

drive_var = collect(150:650) ./ 100.
SKEWNESS = zeros(501)
CVD_STORE = zeros(501)
MEAN_STORE = zeros(501)
# DRIVE=1
for DRIVE=1:450

s1 = drive_var[DRIVE]
s2 = drive_var[DRIVE]

#run simulation
@time t, r = euler_lif_CSR(h, runtime, Ne, W, CSR, s1, s2, input_width, vth, tau_m, tau_s, tau_a, g_a, ang)

# fo="raster_continuum.txt" #filename to save data under
#write_raster: writes spike times to comma-delimited text file
#column 1: time (in iterations), column 2: neuron index
# write_raster(fo, t, r)


###########################
###calculate statistics###
##########################


#prelims
min_e_neurons = 20
min_i_neurons = 50
min_spikes_e = 10
min_spikes_i = 10
fbinsize = 400/h
cbinsize = 100/h
netd_binsize = 50/h

e_m = findall(r .<= Ne);
# i_m = find(r .> Ne);
te = t[e_m];
re = r[e_m];
# ti = t[i_m];
# ri = r[i_m];

er1 = Set(findall(re .< Ne2));
er2 = setdiff(Set(1:length(re)), er1);
# ir1 = Set(find(ri .< Ne + Ni2));
# ir2 = setdiff(Set(1:length(ri)), ir1);

ntd, nts = nt_diff_H(te, re, ntotal, Ne2, netd_binsize);
s = ntd ./ nts ;#signal for dominances
flags, times = WLD_01(s, -.333, .333);


TN, BN = Neurons_tb_ns(re, Ne2, 10, 100); #neurons in either pool who fired at least 10 spkes in simulation
top, tdom, bot, bdom, nmz, tnmz = splice_flags(flags, times, netd_binsize); #find win, lose, and draw times
tbf, rbf = ligase(bot, bdom, te, re, BN); #bottom pool up states
ttf, rtf = ligase(top, tdom, te, re, TN); #top pool up states
tbdf, rbdf = ligase(top, tdom, te, re, BN); #bottom pool down states
ttdf, rtdf = ligase(bot, bdom, te, re, TN); #top pool down states

###main statistics
#T:= pool1
#B:=pool 2
#U: dominance state
#D: suppressed state

#spike-time correlations
cwTu = rand_pair_cor(cbinsize, ttf, rtf, TN, 1000);
cwBu = rand_pair_cor(cbinsize, tbf, rbf, BN, 1000);
cwBd = rand_pair_cor(cbinsize, ttdf, rtdf, TN, 1000);
cwTd = rand_pair_cor(cbinsize, tbdf, rbdf, BN, 1000);

#CVISI
CV_TU = CV_ISI(top, TN, te, re);
CV_BU = CV_ISI(bot, BN, te, re);
CV_BD = CV_ISI(top, BN, tbdf, rbdf);
CV_TD = CV_ISI(bot, TN, ttdf, rtdf);

#dominance statistics
d = convert(Array{Float64}, diff(netd_binsize/(1000. / h) .* times));
cvd = cv(d);

LP = .3;

dx = [];
for i in d
    if i > LP
        push!(dx, i)
    end
end
dx = convert(Array{Float64}, dx);
cvdlp = cv(dx);


MDT, MDB = (tdom*1. / length(top)), (bdom*1. / length(bot));

SKEWNESS[DRIVE] = skewness(dx)
MEAN_STORE[DRIVE] = mean(dx)
CVD_STORE[DRIVE] = cvdlp
end

write_raster("continuum_drive_skewness.txt", drive_var, SKEWNESS)
write_raster("continuum_drive_CVD.txt", drive_var, CVD_STORE)
