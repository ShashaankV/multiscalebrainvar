# Multiscale variability in neuronal competition.

Computer code accompanying manuscript on multiscale variability in neuronal competition. Here we model spiking and perceptual varibiality using a canonical cortical circuit model. Canonical in that it reproduces many basic (but nontrivial) cognitive traits. Cortical circuit in that in that it is based on fundamental neuronal properties such as spike frequency adaptation and reasonable neuron dynamics [may qualify this more]. As shown in manuscript, the discrete mutual inhibition and continuum systems reproduce empirical spiking and perceptual variability statistics without added noise, and other percept dynamics. The empirically-derived statistics are nontrivial and robust to stimulus conditions.

## Credits

Original code written by Benjamin P Cohen (<url>https://github.com/benja-matic</url>). Repository generated and maintained by Shashaank Vattikuti. Please cite this manuscript if you use the code:

<i>Multiscale variability in neuronal competition. Benjamin P Cohen, Carson C Chow, and Shashaank Vattikuti. 	arXiv: [q-bio.NC]</i>

## Prerequisites

Computer code uses Julia v 0.6.0 for simulations. Requires "Distributions" package.

## Data

Main output is raster in comma-delimited text file (spike neuron index and time).<br>
The Python 3 script, plotraster.py followed by &lt;raster filename&gt; can be used plot the results.<br>
Calculations such as CVISI, CVD, and mean dominance duration are included in "run_" scripts for discrete and continuum models.   

## Code for three network architectures

There are three network architectures: 1): unstructured network, 2) discrete mutual inhibition network (between unstructured pools), and 3) structured continuum network. Main codes for running simulations start with "run_".


<!--What things you need to install the software and how to install them -->
