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
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

using namespace dealii;

// Class representing the non-linear diffusion problem.
class DiffusionNonLinear
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 3;

  // Function for the Diffusion tensor coefficient.
  class FunctionD : public Function<dim>
  {
    public:

    FunctionD(const double &d_axn_, const double &d_ext_, const int &axonal_field_)
      : d_axn(d_axn_), d_ext(d_ext_), axonal_field(axonal_field_) {}

    virtual void
    white_tensor_value(const Point<dim> & p,
                 Tensor<2, dim> &values) const
    {
      Tensor<1, dim> n;                  // Axonal direction versor.
      values.clear(); // Clear the tensor values.
      switch (axonal_field){
      case 1: // Isotropic diffusion coefficient
        for(unsigned int i = 0; i < dim; ++i)
          values[i][i] = d_ext;
        break;
      case 2: // Radial axonal diffusion coefficient
        for(unsigned int i = 0; i < dim; ++i)
        {
          n[i] = (p[i] - center[i]);
        }
        n /= (n.norm() + 1e-10); // Normalize the vector
      
        values = outer_product(n, n);

        for(unsigned int i = 0; i < dim; ++i)
        {
          values[i][i] += d_ext;
          for(unsigned int j = 0; j < dim; ++j)
          {
            values[i][j] *= d_axn;
          }
        }
        break;
      case 3: // Circular axonal diffusion coefficient
        n[0] = 0.0;
        n[1] = -(p[2] - center[2]);
        n[2] =  p[1] - center[1];
        n /= (n.norm() + 1e-10); // Normalize the vector

        values = outer_product(n, n);
        for(unsigned int i = 0; i < dim; ++i)
        {
          values[i][i] += d_ext;
          for(unsigned int j = 0; j < dim; ++j)
          {
            values[i][j] *= d_axn;
          }
        }
        break;
      case 4: // Axonal based diffusion coefficient
        if ((p[0] - center[0]) * (p[0] - center[0]) + ((p[1] - center[1]) / 2.0) * ((p[1] - center[1]) / 2.0) + (p[2] - center[2]) * (p[2] - center[2]) < 10.0 * 10.0){
          n[0] = 0.0;
          n[1] = -(p[2] - center[2]);
          n[2] = (p[1] - center[1])/2.0;
          n /= (n.norm() + 1e-10); // Normalize the vector

          values = outer_product(n, n);
          for(unsigned int i = 0; i < dim; ++i)
          {
            values[i][i] += d_ext;
            for(unsigned int j = 0; j < dim; ++j)
            {
              values[i][j] *= d_axn;
            }
          }
        }else{
          for(unsigned int i = 0; i < dim; ++i)
          {
            n[i] = (p[i] - center[i]);
          }
          n /= (n.norm() + 1e-10); // Normalize the vector
        
          values = outer_product(n, n);

          for(unsigned int i = 0; i < dim; ++i)
          {
            values[i][i] += d_ext;
            for(unsigned int j = 0; j < dim; ++j)
            {
              values[i][j] *= d_axn;
            }
          }
        }
        break;
      default:
        AssertThrow(false, ExcMessage("Invalid axonal field type."));
      }
    }

    virtual void
    gray_tensor_value(const Point<dim> & /*p*/,
                 Tensor<2, dim> &values) const
    {
      values.clear(); 
      for(unsigned int i = 0; i < dim; ++i)
          values[i][i] = d_ext;
    }

    virtual double
    white_value(const Point<dim> & p,
          const unsigned int col = 0,
          const unsigned int row = 0) const
    {
      Tensor<1, dim> n;
      switch (axonal_field) {
        case 1: // Isotropic diffusion coefficient
          return d_ext * (col == row ? 1 : 0);
          break;
        case 2: // Radial axonal diffusion coefficient
          for(unsigned int i = 0; i < dim; ++i) {
            n[i] = p[i] - center[i];
          }
          n /= (n.norm() + 1e-10); // Normalize the vector
          return outer_product(n, n)[row][col] * d_axn + d_ext * (col == row ? 1 : 0);
          break;
        case 3: // Circular axonal diffusion coefficient
          n[0] = 0.0;
          n[1] = -(p[2] - center[2]);
          n[2] =  p[1] - center[1];
          n /= (n.norm() + 1e-10); // Normalize the vector
          return outer_product(n, n)[row][col] * d_axn + d_ext * (col == row ? 1 : 0);
          break;
        case 4: // Axonal based diffusion coefficient
          if ((p[0] - center[0]) * (p[0] - center[0]) + ((p[1] - center[1]) / 2.0) * ((p[1] - center[1]) / 2.0) + (p[2] - center[2]) * (p[2] - center[2]) < 10.0 * 10.0){
            n[0] = 0.0;
            n[1] = -(p[2] - center[2]);
            n[2] =  (p[1] - center[1])/2.0;
            n /= (n.norm() + 1e-10); // Normalize the vector

            return outer_product(n, n)[row][col] * d_axn + d_ext * (col == row ? 1 : 0);
          }else{
            for(unsigned int i = 0; i < dim; ++i)
            {
              n[i] = (p[i] - center[i]);
            }
            n /= (n.norm() + 1e-10); // Normalize the vector
          
            return outer_product(n, n)[row][col] * d_axn + d_ext * (col == row ? 1 : 0);
          }
          break;
        default:
          AssertThrow(false, ExcMessage("Invalid axonal field type."));
      }
    }

    virtual double
    gray_value(const Point<dim> & /*p*/,
          const unsigned int col = 0,
          const unsigned int row = 0) const
    {
      return d_ext * (col == row ? 1 : 0); 
    }

    protected: 
    const double d_ext;
    const double d_axn;
    const int axonal_field; // Axonal field type (1: radial, 2: circular, 3: axonal)   
  };

  // Function for the alpha coefficient.
  class FunctionAlpha : public Function<dim>
  {
  public:

    FunctionAlpha(const double &alp_)
      : alp(alp_) {}

    virtual double
    white_value(const Point<dim> & /*p*/ = Point<dim>(),
          const unsigned int /*component*/ = 0) const
    {
      return alp;
    }

    virtual double
    gray_value(const Point<dim> & /*p*/ = Point<dim>(),
          const unsigned int /*component*/ = 0) const
    {
      return alp / 2.0;
    }
  
  protected:
    const double alp;
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

    FunctionU0(const double &protein_type_)
      : protein_type(protein_type_) {}
    virtual double
    value(const Point<dim> & p,
          const unsigned int /*component*/ = 0) const override
    {
      switch (protein_type) {
        case 1:{ // Amyloid-beta initial condition (two spheres and a paraboloid)
          double z = -8.0/605.0 * p[1] * p[1] + 2.0 * p[1] + 25.62; // Get point of the parabola
          if ((p[0] - 50.0) * (p[0] - 50.0) + (p[1] - 40.0) * (p[1] - 40.0) + (p[2] - 50.0) * (p[2] - 50.0) < 20.0*20.0 ||
              ((p[0] - 50.0) * (p[0] - 50.0) + (p[2] - z) * (p[2] - z) < 30.0*30.0 &&
              (p[0] - 60.0) * (p[0] - 60.0) + (p[1] - 95.0) * (p[1] - 95.0) + (p[2] - 110.0) * (p[2] - 110.0) > 20.0*20.0))
            return 0.1;
          else
            return 1e-6; // Small value to avoid negative values in the solution
          break;}
        case 2: // Tau initial condition (a sphere)
          if ((p[0] - 50.0) * (p[0] - 50.0) + (p[1] - 90.0) * (p[1] - 90.0) + (p[2] - 60.0) * (p[2] - 60.0) < 5.0*5.0)
            return 0.2;
          else
            return 1e-6; // Small value to avoid negative values in the solution
          break;
        case 3: // Alpha-Synuclein initial condition (a cylinder with cuts)
          if ((p[1] - 95.0) * (p[1] - 95.0) + (p[2] - 20.0) * (p[2] - 20.0) < 30.0*30.0 &&
              p[0] > 40.0 && p[0] < 65.0 &&
              p[1] < 95.0 &&
              p[2] > -1.5 * p[1] + 155.0)
            return 0.2;
          else
            return 1e-6; // Small value to avoid negative values in the solution
          break;
        case 4: // TDP-43 initial condition (a parallelipiped and a cylinder with cuts)
          if ((p[0] > 45.0 && p[0] < 70.0 &&
               p[1] > 55.0 && p[1] < 75.0 &&
               p[2] > 80.0) ||
              ((p[1] - 95.0) * (p[1] - 95.0) + (p[2] - 20.0) * (p[2] - 20.0) < 15.0*15.0 &&
               p[0] > 45.0 && p[0] < 60.0 &&
               p[1] < 95.0 &&
               p[2] > -1.5 * p[1] + 155.0))
            return 0.15;
          else
            return 1e-6; // Small value to avoid negative values in the solution
          break;
        default:
          AssertThrow(false, ExcMessage("Invalid protein type."));
      }
    }
    private:
    const int protein_type;
  };
  
  // Constructor. We provide the final time, time step Delta t and theta method
  // parameter as constructor arguments.
  DiffusionNonLinear(const std::string  &mesh_file_name_,
                const unsigned int &r_,
                const double       &T_,
                const double       &deltat_,
                const double       &theta_,
                const int          &matter_type_,
                const double       &d_axn_,
                const double       &d_ext_,
                const double       &alp_,
                const int          &protein_type_,
                const int          &axonal_field_,
              )
    : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank == 0)
    , T(T_)
    , mesh_file_name(mesh_file_name_)
    , r(r_)
    , deltat(deltat_)
    , theta(theta_)
    , mesh(MPI_COMM_WORLD)
    , D(d_axn_, d_ext_, axonal_field_)
    , alpha(alp_)
    , u_0(protein_type_)
    , matter_type(matter_type_)
  {}

  // Initialization.
  void
  setup(const Point<dim> &center = Point<dim>());

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

  // Brain matter type (0: isotropic, 1: white/gray matter).
  const int matter_type;

  // Center of the brain.
  static Point<dim> center; // Center of the brain

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