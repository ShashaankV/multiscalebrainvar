println("This code simulates two mutually inhibitting random networks and compares theoretical firing rate predicitons to simualtions.
It does a coarse parameter sweep and for each set of parameters it stores simulated firing rates and calculates theoretical parameters.
Then it uses theory parameters and equations to predict firing rates in the network.\n")

include("discreteMI.jl")
include("Analyze.jl")

using SpecialFunctions
using Random
using Statistics
using LinearAlgebra
using PyPlot
Random.seed!(21344)

#simulation firing rates
ERS1 = []
ERS2 = []
IRS1 = []
IRS2 = []

#theory parameters
WEE1 = []
WEI1 = []
WIE1 = []
WIEL1 = []
WII1 = []

#inputs to each population
IE1 = []
IE2 = []
II1 = []
II2 = []

#neuron parameters
vth = 20
tau_m = 20.
tau_s = 2.

#external input parameters
s_strength = 5.
fe1 = s_strength
# fe2 = fe1 + 1.#large offset to see if theory works for whacky feedforward inputs
fe2 = fe1 + .01#minimal offset makes the singularity in theory rates visible, otherwise you get 0/0
fi1 = 0.#1.2
fi2 = fi1
# fi2 = fi1 + .4#large offset to show that theory works for 4 different feedforward inputs

#number of neurons in each popualtion
N = 4000
k=200
ks = sqrt(k)
IFRAC = 2.
N2 = Int64(round(N/2))
NiL = Int64(round(N2/IFRAC))
NeL = Int64(N2-NiL)
Ne2 = NeL*2
Ni2 = NiL*2

#time
runtime = 10000 #ms
h = .1 #time step
ntotal = round(runtime/h) #time points
rt = (ntotal/1000.)*h #simulation time in seconds

#Scanning long-range connections
Aie_NLS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]

for increment = 1:length(Aie_NLS)
  #weights matrix parameters
  Aee = 10.5
  Aei = 30.
  Aie = 40.
  Aie_NL = Aie_NLS[increment]
  Aii = 35

#compute weights matrix and sparse representation
W = homogenous_4x4_weights(N, IFRAC, k, Aee, Aei, Aie, Aie_NL, Aii);
CSR = sparse_rep(W, N);
#run simulation and analyze raster (this is a fast verions of the code with no adaptation variables, no measured inputs, etc.)
@time t, r = molda_euler_lif_CSR_noa(h, runtime, N, IFRAC, W, CSR, fe1, fi1, fe2, fi2, vth, tau_m, tau_s)

#chunk the raster into each population
e_m = findall(r .<= Ne2);
i_m = findall(r .> Ne2);
te = t[e_m];
re = r[e_m];
ti = t[i_m];
ri = r[i_m];
#normalization factor to firing rate, E and I populations have the same size so you can use one factor for both
nmz = rt * NeL
nmz_ms = runtime * NeL
#number of spikes from each population
nse1 = length(findall(NeL+1 .<= re .<= Ne2))
nse2 = length(findall(1 .<= re .<= NeL))
nsi1 = length(findall(N2 + NiL+1 .<= ri .<= N2 + Ni2))
nsi2 = length(findall(N2 + 1 .<= ri .<= N2 + NiL))
#use number of spikes to back-calculate the mean input in each population
mie1 = (((nse1 * tau_s * ks * Aee) - (nsi1 * tau_s * ks * Aei))/nmz_ms) + fe1
mie2 = (((nse2 * tau_s * ks * Aee) - (nsi2 * tau_s * ks * Aei))/nmz_ms) + fe2
mii1 = (((nse1 * tau_s * ks * Aie) + (nse2 * tau_s * ks * Aie_NL)- (nsi1 * tau_s * ks * Aii))/nmz_ms) + fi1
mii2 = (((nse2 * tau_s * ks * Aie) + (nse1 * tau_s * ks * Aie_NL)- (nsi2 * tau_s * ks * Aii))/nmz_ms) + fi2
#store inputs
push!(IE1, mie1)
push!(IE2, mie2)
push!(II1, mii1)
push!(II2, mii2)
#use number of spikes to calcualte firing rate
MER1 = nse1/nmz
MER2 = nse2/nmz
MIR1 = nsi1/nmz
MIR2 = nsi2/nmz
#store firing rates
push!(ERS1, MER1)
push!(ERS2, MER2)
push!(IRS1, MIR1)
push!(IRS2, MIR2)
#calculate theoretical parameters (virtually identical results can be obtained by solving w_ij = input_ij / r_j)
WEE = Aee * ks * tau_s / 1000.
WEI = Aei * ks * tau_s / 1000.
WIE = Aie * ks * tau_s / 1000.
WIEL = Aie_NLS[increment] * ks * tau_s / 1000.
WII = Aii * ks * tau_s / 1000.
#store theory parameters
push!(WEE1, WEE)
push!(WEI1, WEI)
push!(WIE1, WIE)
push!(WIEL1, WIEL)
push!(WII1, WII)
end

#classify dominant vs. suppressed populations with firing rates
big_inh = zeros(length(ERS1)) #inhibitory rate in dominant CIRCUIT
big_exc = zeros(length(ERS1)) #excitatory rate in dominant CIRCUIT
lil_inh = zeros(length(ERS1)) #same, for suppressed circuits
lil_exc = zeros(length(ERS1))
for i = 1:length(ERS1)
 if ERS1[i] > ERS2[i]
  big_inh[i] = IRS1[i]
  big_exc[i] = ERS1[i]
  lil_inh[i] = IRS2[i]
  lil_exc[i] = ERS2[i]
 else
  big_inh[i] = IRS2[i]
  big_exc[i] = ERS2[i]
  lil_inh[i] = IRS1[i]
  lil_exc[i] = ERS1[i]
 end
end

#separate out which input belongs to which
big_inh_input = zeros(length(ERS1))
big_exc_input = zeros(length(ERS1))
lil_inh_input = zeros(length(ERS1))
lil_exc_input = zeros(length(ERS1))
for i = 1:length(ERS1)
 if ERS1[i] > ERS2[i]
  big_inh_input[i] = II1[i]
  big_exc_input[i] = IE1[i]
  lil_inh_input[i] = II2[i]
  lil_exc_input[i] = IE2[i]
 else
  big_inh_input[i] = II2[i]
  big_exc_input[i] = IE2[i]
  lil_inh_input[i] = II1[i]
  lil_exc_input[i] = IE1[i]
 end
end
#Theory equations use + and - signs but weights have absolute values, so use abs
WEEz = [abs(i) for i in WEE1];
WEIz = [abs(i) for i in WEI1];
WIEz = [abs(i) for i in WIE1];
WIELz = [abs(i) for i in WIEL1];
WIIz = [abs(i) for i in WII1];
#calculate optimal threshold with a grid search
et = range(-3, stop = 3, length = 100);
it = range(-3, stop = 3, length = 100);
eidax = zeros(5, 10000);

let c=1
    for i = 1:length(et)
      for j = 1:length(it)
        #find optimal threshold by finding where 2x2 fits, apply to 4x4
        # RETxt, RITxt = theory_rates_2x2(WEEz[10:end], WIEz[10:end], WEIz[10:end], WIIz[10:end], fe1 .- et[i], fi1 .- it[j])
        # e_error = norm(RETxt - big_exc[10:end])
        # i_error = norm(RITxt - big_inh[10:end])
        #find optimal threshold by minimizing error inthe 4x4 region
        RET1xt, RET2xt, RIT1xt, RIT2xt = theory_rates_4x4_sf(WEEz[1:5], WIEz[1:5], WIELz[1:5], WEIz[1:5], WIIz[1:5], fe1 .- et[i], fi1 .- it[j], fe2 .- et[i], fi2 .- it[j])
        e_error = norm(RET1xt - big_exc[1:5])
        i_error = norm(RIT1xt - big_inh[1:5])
        t_error = e_error + i_error
        eidax[1,c] = et[i]
        eidax[2,c] = it[j]
        eidax[3,c] = e_error
        eidax[4,c] = i_error
        eidax[5,c] = t_error
        c+=1
      end
    end
end

x = argmin(eidax[5,:])
etx = eidax[1,x]
itx = eidax[2,x]

TE1 = [etx for i = 1:length(IRS1)];
TI1 = [itx for i = 1:length(IRS1)];
TE2 = [etx for i = 1:length(IRS1)];
TI2 = [itx for i = 1:length(IRS1)];

RE1_4x4_sf, RE2_4x4_sf, RI1_4x4_sf, RI2_4x4_sf = theory_rates_4x4_sf(WEEz, WIEz, WIELz, WEIz, WIIz, fe1 .- TE1, fi1 .- TI1, fe2 .- TE2, fi2 .- TI2);
LE1W, LI1W = theory_rates_2x2(abs(WEE1[1]), abs(WIE1[1]), abs(WEI1[1]), abs(WII1[1]), fe2 .- TE1, fi2 .- TI1);
LE1L, LI1L = theory_rates_2x2(abs(WEE1[1]), abs(WIE1[1]), abs(WEI1[1]), abs(WII1[1]), fe1 .- TE1, fi1 .- TI1);

TE1d = [etx for i = 1:350];
TI1d = [itx for i = 1:350];
TE2d = [etx for i = 1:350];
TI2d = [itx for i = 1:350];
#dense scan so you can see the singularity
WEEzd = [WEE1[1] for i=1:350];
WIEzd = [WIE1[1] for i=1:350];
WIELzd = range(WIEL1[1], stop = WIEL1[end], length = 350);
WEIzd = [WEI1[1] for i=1:350];
WIIzd = [WII1[1] for i=1:350];

RE1_4x4_sf_dense, RE2_4x4_sf_dense, RI1_4x4_sf_dense, RI2_4x4_sf_dense = theory_rates_4x4_sf(WEEzd, WIEzd, WIELzd, WEIzd, WIIzd, fe1 .- TE1d, fi1 .- TI1d, fe2 .- TE2d, fi2 .- TI2d);
WIELzdx = 1000 .* WIELzd ./ (tau_s .* sqrt(k));

# write_array("Molda_EW_sim.txt", big_exc)
# write_array("Molda_EL_sim.txt", lil_exc)
# write_array("Molda_IW_sim.txt", big_inh)
# write_array("Molda_IL_sim.txt", lil_inh)
#
# write_array("Molda_ewi.txt", big_exc_input)
# write_array("Molda_eli.txt", lil_exc_input)
# write_array("Molda_iwi.txt", big_inh_input)
# write_array("Molda_ili.txt", lil_inh_input)
#
# write_array("Molda_E1_th4.txt", RE1_4x4_sf)
# write_array("Molda_E2_th4.txt", RE2_4x4_sf)
# write_array("Molda_I1_th4.txt", RI1_4x4_sf)
# write_array("Molda_I2_th4.txt", RI2_4x4_sf)
#
# write_array("Molda_E1_th2.txt", LE1W)
# write_array("Molda_E2_th2.txt", LE1L)
# write_array("Molda_I1_th2.txt", LI1W)
# write_array("Molda_I2_th2.txt", LI1L)
#
# write_array("Molda_EW_th4d.txt", RE1_4x4_sf_dense)
# write_array("Molda_IW_th4d.txt", RI1_4x4_sf_dense)
# write_array("Molda_EW_th4d_params.txt", WIELzdx)


subplot(221)
plot(WIELzdx, RE1_4x4_sf_dense, ".", ms = 2., label = "Theory")
plot(Aie_NLS, big_exc, "b.", ms = 10., label = "E Win Sim")
plot(Aie_NLS, RE2_4x4_sf, "b+", ms = 10, label = "E Theory win")
plot(Aie_NLS, LE1W, "bx", ms = 10, label = "E_winner 2x2 theory")
legend()
subplot(222)
plot(WIELzdx, RI1_4x4_sf_dense, ".", ms = 2., label = "Theory")
plot(Aie_NLS, big_inh, "b.", ms = 10., label = "I Win Sim")
plot(Aie_NLS, RI2_4x4_sf, "b+", ms = 10, label = "I Theory win")
plot(Aie_NLS, LI1W, "bx", ms = 10, label = "I_winner 2x2 theory")
legend()
subplot(223)
plot(WIELzdx, RE2_4x4_sf_dense, ".", ms = 2., label = "Theory")
plot(Aie_NLS, lil_exc, "b.", ms = 10., label = "E Los Sim")
plot(Aie_NLS, RE1_4x4_sf, "b+", ms = 10, label = "E Theory Los")
plot(Aie_NLS, LE1L, "bx", ms = 10, label = "E Los 2x2 theory")
legend()
subplot(224)
plot(WIELzdx, RI2_4x4_sf_dense, ".", ms = 2., label = "Theory")
plot(Aie_NLS, lil_inh, "b.", ms = 10., label = "I Los Sim")
plot(Aie_NLS, RI1_4x4_sf, "b+", ms = 10, label = "I Theory Los")
plot(Aie_NLS, LI1L, "bx", ms = 10, label = "I Los 2x2 theory")
legend()


#Visualize the 0/0
d = (((WEE1[1] .* WII1[1] ./ WEI1[1]) .- WIE1[1]) .^2.) .- WIELzd .^ 2.
n = (WIELzd .* fi1) .- (WIELzd .* fe1 .* WII1[1] ./ WEI1[1]) .+ ((fi1 .- (fe1 .* WII1[1] ./ WEI1[1])) .* ((WEE1[1] .* WII1[1] ./ WEI1[1]) .- WIE1[1]))

da = (((WEE1[1] .* WII1[1] ./ WEI1[1]) .- WIE1[1]) .^2.) .- WIELzd .^ 2.
na = (WIELzd .* fi2) .- (WIELzd .* fe2 .* WII1[1] ./ WEI1[1]) .+ ((fi1 .- (fe1 .* WII1[1] ./ WEI1[1])) .* ((WEE1[1] .* WII1[1] ./ WEI1[1]) .- WIE1[1]))
figure(2)
x = na ./ da
subplot(221)
plot(d, ".")
plot(n, ".")
subplot(222)
plot(da, ".")
plot(na, ".")
subplot(223)
plot(n ./ d, ".")
subplot(224)
plot(na ./ da, ".")

write_array("T_denom_raw.txt", d)
write_array("T_num_raw.txt", n)

write_array("T_denom_raw_adj.txt", da)
write_array("T_num_raw_adj.txt", na)

write_array("T_frac_raw.txt", d ./ n)
write_array("T_frac_adj.txt", da ./ na)
