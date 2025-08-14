#include "DiffusionNonLinear.hpp"
#include "parameters.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const Point<3> center(55.0, 80.0, 65.0); // Center of the brain

  // TODO: implement multiple problems
  // (maybe read all params at once returning a vector, then in a for loop call problem(p[i]...))
  Parameters p = read_params_from_csv(argv[1]);
  
  DiffusionNonLinear problem(
    p.mesh_file_name,
    p.degree,
    p.T,
    p.deltat,
    p.theta,
    p.matter_type,  // 1: Isotropic, 2: White/Gray
    p.protein_type, // 1: Amyloid-beta, 2: Tau, 3: Alpha-synuclein, 4: TDP-43
    p.axonal_field, // 1: Isotropic, 2: radial, 3: circular, 4: axonal
    p.d_axn,
    p.d_ext,
    p.alpha
  ); // TODO: implement output_dir, to be passed to the output function

  problem.setup(center);
  problem.solve();

  return 0;
}