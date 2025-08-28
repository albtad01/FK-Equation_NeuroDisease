### Organizing the source code
Please place all your sources into the `src` folder.

Binary files must not be uploaded to the repository (including executables).

Mesh files should not be uploaded to the repository. If applicable, upload `gmsh` scripts with suitable instructions to generate the meshes (and ideally a Makefile that runs those instructions). If not applicable, consider uploading the meshes to a different file sharing service, and providing a download link as part of the building and running instructions.

### Mesh processing
The mesh for the brain is in stl format, to convert it to msh run the following:
```bash
$ cd mesh
$ gmsh brain.geo -3 -format msh2 -o brain.msh
$ cd ..
```

### Compiling
To build the executable, make sure you have loaded the needed modules with
```bash
module load gcc-glibc dealii
```
Then run the following commands:
```bash
mkdir build
cd build
cmake ..
make
```
The executable will be created into `build`, and can be executed through
```bash
mpirun -n <N_processes> ./FisherKolmogorov ../src/parameters.csv
```
the flag `--use-hwthread-cpus` can be added to use hardware threads as additional processes.

The parameters.csv file contains all the necessary parameters for the simulation and can be used to run multiple problems at once. For example a parameters.csv file like:
```bash
mesh_file_name,degree,T,deltat,theta,matter_type,protein_type,axonal_field,d_axn,d_ext,alpha,output_dir
../mesh/brain.msh,2,20.0,0.25,1.0,0,1,2,10.0,5.0,0.25,amyloid
../mesh/brain.msh,2,20.0,0.25,1.0,1,4,3,10.0,5.0,0.25,tdp43
```
will run two simulations, one for each line after the header.
The first runs an isotropic simulation studying amyloid-beta deposits with radial axons, while the second runs an  anisotropic (white/gray matter) simulation studying TDP-43 inclusions with circumferential axons.

An example of the output of the simulation visualized in paraview is:
![Simulation Output](media/TDP.gif)