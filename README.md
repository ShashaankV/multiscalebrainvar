# Multiscale variability in neuronal competition.

Computer code accompanying manuscript on multiscale variability in neuronal competition. Here we model spiking and perceptual varibiality using a canonical cortical circuit model. Canonical in that it reproduces many basic (but nontrivial) cognitive traits. Cortical circuit in that in that it is based on fundamental neuronal properties such as spike frequency adaptation and reasonable neuron dynamics [may qualify this more]. As shown in manuscript, the discrete mutual inhibition and continuum systems reproduce empirical spiking and perceptual variability statistics without added noise, and other percept dynamics. The empirically-derived statistics are nontrivial and robust to stimulus conditions.

## Credits

Original code written by Benjamin P Cohen (<url>https://github.com/benja-matic</url>). Repository generated and maintained by Shashaank Vattikuti. Please cite this manuscript if you use the code:

<b>Cohen, Benjamin P., Carson C. Chow, and Shashaank Vattikuti. "Dynamical modeling of multi-scale variability in neuronal competition." Communications Biology 2, no. 1 (2019): 1-11.</b>

## Prerequisites

Computer code requires installation of Julia v 0.6.0 or Julia v 1.1 for simulations; the branch 'julia1' includes code adapted to Julia v 1.1. Requires "Distributions" package. Installation time should be minutes for basic Julia and necessary packages.

## Data

Main output is raster in comma-delimited text file (spike neuron index and time).<br>
The Python 3 script, plotraster.py followed by &lt;raster filename&gt; can be used plot the results.<br>
Calculations such as CVISI, CVD, and mean dominance duration are included in "run_" scripts for discrete and continuum models.   

## Code for three network architectures

There are three network architectures: 1): unstructured network, 2) discrete mutual inhibition network (between unstructured pools), and 3) structured continuum network. Main codes for running simulations start with "run_". Base "run_" codes can be used to generate demo data and run analyses. Simulation time depends on the computer system. Demos take less than a minute.

<!--What things you need to install the software and how to install them -->
