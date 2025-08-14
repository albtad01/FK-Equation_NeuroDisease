#include "DiffusionNonLinear.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const unsigned int degree = 2;

  const double T      = std::stod(argv[1]);
  const double deltat = std::stod(argv[2]);
  const int protein_type = 1; //std::stoi(argv[3]); // 1: amyloid-beta, 2: tau, 3: alpha-synuclein, 4: TDP-43
  const int axonal_field = 1; //std::stoi(argv[4]); // 1: isotropic, 2: radial, 3: circular, 4: axonal
  const int matter_type = 0; //std::stoi(argv[5]); // 0: isotropic, 1: white/gray matter
  const double theta  = 1.0;
  const Point<3> center(55.0, 80.0, 65.0); // Center of the brain

  DiffusionNonLinear problem("../mesh/brain.msh", degree, T, deltat, theta);

  problem.setup(protein_type, axonal_field, matter_type, center);
  problem.solve();

  return 0;
}