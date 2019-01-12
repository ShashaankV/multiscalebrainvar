
println("This code simulates two mutually inhibiting random EI networks with adaptation.
Same parameters used for Figure 3 and 4.")

include("discreteMI.jl")
include("analyze.jl")

Aee = 10.5
Aei = 20.
Aie = 30.
Aie_NL = 30.
Aii = 45.

s_strength = 5.
fe1 = s_strength
fe2 = fe1
fi1 = 0.
fi2 = fi1

k = 200

N = 4000
IFRAC = 2.
N_local = Int64(round(N/2)) #divide the network into two EI circuits
Ni_local = Int64(round(N_local/IFRAC))
Ne_local = Int64(N_local-Ni_local)
Ne2 = Ne_local*2
N2 = Int64(round(N/2)) #divide the network into two EI circuits
NiL = Int64(round(N2/IFRAC))
NeL = Int64(N2-NiL)
Ne2 = NeL*2
Ni2 = NiL*2

Ne = div(N, 2)
Ni = div(N, 2)

vth = 20
tau_m = 20.
tau_s = 2.
tau_a = 350.
g_a = 0.44

runtime = 10*1000 #ms
h = .1 #timestep

W = homogenous_4x4_weights(N, IFRAC, k, Aee, Aei, Aie, Aie_NL, Aii);
CSR = sparse_rep(W, N);

@time t, r = molda_euler_lif_CSR(h, runtime, N, IFRAC, W, CSR, fe1, fi1, fe2, fi2, vth, tau_m, tau_s, tau_a, g_a)

fo="raster_discreteMI.txt" #filename to save data under
#write_raster: writes spikes times to comma-delimited text file
#column 1: time (in iterations), column 2: neuron index
write_raster(fo, t, r)

###########################
###calculate statistics###
##########################

###prelims####
e_m = findall(r .<= Ne2);
i_m = findall(r .> Ne2);
te = t[e_m];
re = r[e_m];
ti = t[i_m];
ri = r[i_m];

TN, BN = Neurons_tb_ns(re, NeL, 10, 100) #neurons in either pool who fired at least 10 spkes in simulation

ntd, nts = nt_diff_H(te, re, ntotal, NeL, netd_binsize)

s = ntd ./ nts #signal for dominances

flags, times = WLD_01(s, -.333, .333)

top, tdom, bot, bdom, nmz, tnmz = splice_flags(flags, times, netd_binsize) #find win, lose, and draw times

tbf, rbf = ligase(bot, bdom, te, re, BN) #bottom pool up states

ttf, rtf = ligase(top, tdom, te, re, TN) #top pool up states

tbdf, rbdf = ligase(top, tdom, te, re, BN) #bottom pool down states

ttdf, rtdf = ligase(bot, bdom, te, re, TN) #top pool down states


####main statistics###
#T:= pool1
#B:=pool 2
#U: dominance state
#D: suppressed state

#correlations, vector across sampled neurons
cwTu = rand_pair_cor(cbinsize, ttf, rtf, TN, 1000)
cwBu = rand_pair_cor(cbinsize, tbf, rbf, BN, 1000)
cwBd = rand_pair_cor(cbinsize, ttdf, rtdf, TN, 1000)
cwTd = rand_pair_cor(cbinsize, tbdf, rbdf, BN, 1000)

#CV ISI, vector across sampled neurons
CV_TU = CV_ISI(top, TN, te, re)
CV_BU = CV_ISI(bot, BN, te, re)
CV_BD = CV_ISI(top, BN, tbdf, rbdf)
CV_TD = CV_ISI(bot, TN, ttdf, rtdf)

#dominance durations, d and dx are arrays of dominance times
d = convert(Array{Float64}, diff(netd_binsize/(1000. /h) .* times))
cvd = cv(d)

#report thresholded (LP) dominance statistics
LP = .3

dx = []
for i in d
    if i > LP
        push!(dx, i)
    end
end
dx = convert(Array{Float64}, dx)

cvdlp = cv(dx) #cvd after thresholding

MDT = tdom/length(top) #mean dominance, pool 1
MDB = bdom/length(bot) #mean dominance, pool 2
MDN = tnmz/length(nmz)
