include("continuum.jl")
include("analyze.jl")

println("This code simulates rivalry in the continuum network with adaptation.
Parameters for Figure 3 results.")

using SpecialFunctions
srand(2134)

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
println(kee)
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
s1 = 3.
s2 = 3.
#Time
runtime = 10*1000#ms
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

#run simulation
@time t, r = euler_lif_CSR(h, runtime, Ne, W, CSR, s1, s2, input_width, vth, tau_m, tau_s, tau_a, g_a, ang)

fo="continuum_spikerecord.txt" #filename to save data under
#write_raster: writes spikes times to comma-delimited text file
#column 1: time (in iterations), column 2: neuron index
write_raster(fo, t, r)
