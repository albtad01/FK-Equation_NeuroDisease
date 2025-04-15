#include "HeatNonLinear.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const unsigned int degree = 1;

  const double T      = 1.0;
  const double deltat = 0.05;
  const double theta  = 1.0;

  HeatNonLinear problem("../mesh/mesh-square-20.msh", degree, T, deltat, theta);

  problem.setup();
  problem.solve();

  return 0;
}