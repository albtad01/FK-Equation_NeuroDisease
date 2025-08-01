#include "HeatNonLinear.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const unsigned int degree = 2;

  const double T      = std::stod(argv[1]);
  const double deltat = std::stod(argv[2]);
  const double theta  = 1.0;

  HeatNonLinear problem("../mesh/brain.msh", degree, T, deltat, theta);

  problem.setup();
  problem.solve();

  return 0;
}