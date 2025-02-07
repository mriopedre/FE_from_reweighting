# FE from reweighting

Python code to calculate free energy from reweighting a metadynamics trajectory. Complementary method to plumed sum_hills, usefull sometimes - error analysis, complex CVs, PBMetaD, checking convergence... 
Example of reweight trajectory from plumed in plumed_reweight.dat

To produce the example file run plumed driver to print the bias and the value of any CV of interest in a file.
The example plumed file is provided as plumed_reweight.dat.
The command to execute it in plumed as follows:
	plumed driver --mf_xtc md.xtc --plumed plumed_reweight.dat --kt 2.57 
 
```python
#Example usage
#Input folders and files
#File is the colvar from plumed Driver - see example Driver
#Set KBT according to the temperature of the simulation
fol = '/example_folder/'
file = fol+'/COLVAR_REWEIGHT'
KBT=2.57

#Read the colvar file and split it adequatelly 
#Example here for a given Colvar file. Split accordingly to the field in your Colvar.
time, qz, qy, qx, bias = np.loadtxt(file).T

#Calculate weights of frames using the bias
weights = calculate_weights(KBT, bias)

#Examples to plot the free energy for 1D or 2D CVs
#sum_hills - optional, file - point to file with sum_hills results. Shows comparison.
#plot_histogram - optional, bolean - shows sampling of the CV by printing the unweigthed histogram. 
plot_reweighted_FE_1D(qz, weights, sum_hills = f+'/qz.dat', plot_histogram=True)
#Plot 2D CVs
plot_reweighted_FE_2D(qz, qx, weights)
```
