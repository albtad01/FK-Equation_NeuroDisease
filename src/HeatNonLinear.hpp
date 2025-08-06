#ifndef HEAT_NON_LINEAR_HPP
#define HEAT_NON_LINEAR_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

using namespace dealii;

// Class representing the non-linear diffusion problem.
class HeatNonLinear
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 3;

  // Function for the Diffusion tensor coefficient.
  class FunctionD : public Function<dim>
  {
    // TODO DIFFERENCIATE BETWEEN WHITE AND GRAY MATTER WITH DIFFERENT FUNCTIONS
    public:
    virtual void
    tensor_value(const Point<dim> & p,
                 Tensor<2, dim> &values) const
    {
      Tensor<1, dim> n;                  // Axonal direction versor.
      switch (axonal_field){
      case 1: // TODO implement real function
        for(unsigned int i = 0; i < dim; ++i)
        {
          n[i] = p[i]/p.norm(); // Value might change when mesh is distributed?
        }
        
        values = outer_product(n, n);

        for(unsigned int i = 0; i < dim; ++i)
        {
          for(unsigned int j = 0; j < dim; ++j)
          {
            values[i][j] *= d_axn;
          }
        }
        
        for(unsigned int i = 0; i < dim; ++i)
        {
          values[i][i] += d_ext;
        }
        break;
      case 2:
        // TODO Circular axonal diffusion coefficient
        break;
      case 3:
        // TODO Axonal based diffusion coefficient
        break;
      default:
        AssertThrow(false, ExcMessage("Invalid axonal field type."));
      }
    }

    virtual double
    value(const Point<dim> & p,
          const unsigned int col = 0,
          const unsigned int row = 0) const
    {
      Tensor<1, dim> n;
      switch (axonal_field) {
        case 1: // TODO implement real function
          for(unsigned int i = 0; i < dim; ++i) {
            n[i] = p[i]/p.norm();
          }
          return outer_product(n, n)[col][row] * d_axn + d_ext * (col == row ? 1 : 0);
          break;
        case 2:
          // TODO Circular axonal diffusion coefficient
          break;
        case 3:
          // TODO Axonal based diffusion coefficient
          break;
        default:
          AssertThrow(false, ExcMessage("Invalid axonal field type."));
      }
    }

    protected: // TODO see if gray necessary or not
    const double d_ext = 0.0005;
    const double d_ext_gray = 0.0005; // May be unnecessary, gray matter just uses d_ext only?
    const double d_axn = 0.001;
    const double d_axn_gray = 0.001; // May be unnecessary, white matter just uses d_axn in addition?
  };

  // Function for the alpha coefficient.
  class FunctionAlpha : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & /*p*/ = Point<dim>(),
          const unsigned int /*component*/ = 0) const override
    {
      return alp;
    }
  
  protected: // TODO see if gray necessary or not
    const double alp = 1.0;
    const double alp_gray = 0.5;
  };

  // Function for the forcing term.
  class ForcingTerm : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return 0.0;
    }
  };

  // Function for initial conditions.
  class FunctionU0 : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & p,
          const unsigned int /*component*/ = 0) const override
    {
      switch (protein_type) {
        case 1:
          // TODO Amyloid-beta initial condition
          if ((p[0] - 0.7) * (p[0] - 0.7) + (p[1] - 0.8) * (p[1] - 0.8) + (p[2] - 0.7) * (p[2] - 0.7) < 0.05*0.05)
            return 0.5;
          else
            return 1e-6; // Small value to avoid negative values in the solution
          break;
        case 2:
          // TODO Tau initial condition
          if ((p[0] - 0.7) * (p[0] - 0.7) + (p[1] - 0.8) * (p[1] - 0.8) + (p[2] - 0.7) * (p[2] - 0.7) < 0.05*0.05)
            return 0.5;
          else
            return 1e-6; // Small value to avoid negative values in the solution
          break;
        case 3:
          // TODO Alpha-Synuclein initial condition
          if ((p[0] - 0.7) * (p[0] - 0.7) + (p[1] - 0.8) * (p[1] - 0.8) + (p[2] - 0.7) * (p[2] - 0.7) < 0.05*0.05)
            return 0.5;
          else
            return 1e-6; // Small value to avoid negative values in the solution
          break;
        case 4:
          // TODO TDP-43 initial condition
          if ((p[0] - 0.7) * (p[0] - 0.7) + (p[1] - 0.8) * (p[1] - 0.8) + (p[2] - 0.7) * (p[2] - 0.7) < 0.05*0.05)
            return 0.5;
          else
            return 1e-6; // Small value to avoid negative values in the solution
          break;
        default:
          AssertThrow(false, ExcMessage("Invalid protein type."));
      }
    }
  };

  // Constructor. We provide the final time, time step Delta t and theta method
  // parameter as constructor arguments.
  HeatNonLinear(const std::string  &mesh_file_name_,
                const unsigned int &r_,
                const double       &T_,
                const double       &deltat_,
                const double       &theta_)
    : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank == 0)
    , T(T_)
    , mesh_file_name(mesh_file_name_)
    , r(r_)
    , deltat(deltat_)
    , theta(theta_)
    , mesh(MPI_COMM_WORLD)
  {}

  // Initialization.
  void
  setup(const int &protein_type_,
        const int &axonal_field_,
        const int &matter_type_);

  // Solve the problem.
  void
  solve();

protected:
  // Assemble the tangent problem.
  void
  assemble_system();

  // Solve the linear system associated to the tangent problem.
  void
  solve_linear_system();

  // Solve the problem for one time step using Newton's method.
  void
  solve_newton();

  // Output.
  void
  output(const unsigned int &time_step) const;

  // MPI parallel. /////////////////////////////////////////////////////////////

  // Number of MPI processes.
  const unsigned int mpi_size;

  // This MPI process.
  const unsigned int mpi_rank;

  // Parallel output stream.
  ConditionalOStream pcout;

  // Problem definition. ///////////////////////////////////////////////////////

  // mu_0 coefficient.
  FunctionD D;

  // mu_1 coefficient.
  FunctionAlpha alpha;

  // Forcing term.
  ForcingTerm forcing_term;

  // Initial conditions.
  FunctionU0 u_0;

  // Current time.
  double time;

  // Final time.
  const double T;

  // Discretization. ///////////////////////////////////////////////////////////

  // Mesh file name.
  const std::string mesh_file_name;

  // Polynomial degree.
  const unsigned int r;

  // Time step.
  const double deltat;

  // Theta parameter of the theta method.
  const double theta;

  // Protein type (1: amyloid-beta, 2: tau, 3: alpha-synuclein, 4: TDP-43).
  static int protein_type;

  // Axonal field type (1: radial, 2: circular, 3: axonal).
  static int axonal_field;

  // Brain matter type (0: isotropic, 1: white/gray matter).
  static int matter_type;

  // Mesh.
  parallel::fullydistributed::Triangulation<dim> mesh;

  // Finite element space.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature formula.
  std::unique_ptr<Quadrature<dim>> quadrature;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // DoFs owned by current process.
  IndexSet locally_owned_dofs;

  // DoFs relevant to the current process (including ghost DoFs).
  IndexSet locally_relevant_dofs;

  // Jacobian matrix.
  TrilinosWrappers::SparseMatrix jacobian_matrix;

  // Residual vector.
  TrilinosWrappers::MPI::Vector residual_vector;

  // Increment of the solution between Newton iterations.
  TrilinosWrappers::MPI::Vector delta_owned;

  // System solution (without ghost elements).
  TrilinosWrappers::MPI::Vector solution_owned;

  // System solution (including ghost elements).
  TrilinosWrappers::MPI::Vector solution;

  // System solution at previous time step.
  TrilinosWrappers::MPI::Vector solution_old;
};

#endif