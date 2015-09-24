/*
 *    An MIT forward solver code based on the deal.II (www.dealii.org) library.
 *    Copyright (C) 2013-2015 Ross Kynch & Paul Ledger, Swansea Unversity.
 * 
 *    This program is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program.  If not, see <http://www.gnu.org/licenses/>.    
*/

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>

#include <deal.II/grid/manifold_lib.h>

#include <all_data.h>
#include <backgroundfield.h>
#include <curlfunction.h>
#include <forwardsolver.h>
#include <inputtools.h>
#include <mydofrenumbering.h>
#include <mypreconditioner.h>
#include <myvectortools.h>
#include <outputtools.h>

#include <myfe_nedelec.h>

using namespace dealii;

namespace torusBenchmark
{
  
  template <int dim>
  class torusBenchmark
  {
  public:
    torusBenchmark (const unsigned int poly_order,
                     const unsigned int mapping_order = 2);
    ~torusBenchmark ();
    void run(std::string input_filename,
             std::string output_filename);
  private:
    Triangulation<dim> tria;
    FESystem<dim> fe;
    DoFHandler<dim> dof_handler;

    const unsigned int poly_order;
    const unsigned int mapping_order;
    
    void initialise_materials();
    void process_mesh(bool neuman_flag);
  };
  
  template <int dim>
  torusBenchmark<dim>::torusBenchmark(const unsigned int poly_order,
                                        const unsigned int mapping_order)
  :
  fe (MyFE_Nedelec<dim>(poly_order), 2),
  dof_handler (tria),
  poly_order(poly_order),
  mapping_order(mapping_order)
  {
  }
  
  template <int dim>
  torusBenchmark<dim>::~torusBenchmark ()
  {
    dof_handler.clear ();  
  }
  
  template <int dim>
  void torusBenchmark<dim>::initialise_materials()
  {
    EquationData::param_mur.reinit(2);
    EquationData::param_mur(0) = EquationData::param_mur_background;       
    EquationData::param_mur(1) = EquationData::param_mur_conducting;
    
    EquationData::param_sigma.reinit(2);
    EquationData::param_sigma(0) = EquationData::param_sigma_background;
    EquationData::param_sigma(1) = EquationData::param_sigma_conducting;
    
    EquationData::param_epsilon.reinit(2);
    EquationData::param_epsilon(0) = EquationData::param_epsilon_background;
    EquationData::param_epsilon(1) = EquationData::param_epsilon_conducting;
    
    
    // kappa = (-omega^2*epsilon + i*omega*sigma)*mu_0;
    // i.e. kappa_re = -mu_0*epsilon*omega^2
    //      kappa_im = mu_0*omega*sigma
    EquationData::param_kappa_re.reinit(EquationData::param_mur.size());
    EquationData::param_kappa_im.reinit(EquationData::param_mur.size());
    for (unsigned int i=0; i<EquationData::param_mur.size(); ++i) // note mur and kappa must have same size:
    {
      // Unregularised - will be handled in forward solver.
      EquationData::param_kappa_re(i) = 0.0;
      EquationData::param_kappa_im(i) = EquationData::param_sigma(i)*EquationData::param_omega*EquationData::constant_mu0;
      // TODO:
      // remove this later.
      // This was how I handled it, which would regularised all DoFs if they're in a cell with zero sigma (and thus zero kappa).
      /*
      if (EquationData::param_sigma(i) > 0)
      {
        EquationData::param_kappa_re(i) = 0.0;//EquationData::param_omega*EquationData::param_sigma(i)*EquationData::constant_mu0;
        EquationData::param_kappa_im(i) = EquationData::param_sigma(i)*EquationData::param_omega*EquationData::constant_mu0;
      }
      else
      {
        EquationData::param_kappa_re(i) = 0.0;//EquationData::param_regularisation*EquationData::param_omega*EquationData::constant_mu0;
        EquationData::param_kappa_im(i) = EquationData::param_regularisation*EquationData::param_omega*EquationData::constant_mu0;
        //0.0;//EquationData::param_regularisation*EquationData::constant_mu0*EquationData::param_omega;
      }*/
    }    
  }
  template <int dim>
  void torusBenchmark<dim>::process_mesh(bool neumann_flag)
  {
    // Routine to process the read in mesh
    typename Triangulation<dim>::cell_iterator cell;
    const typename Triangulation<dim>::cell_iterator endc = tria.end();
    
    
    // Find faces shared between cells within the conductor
    // and outside the conductor:
    // First mark all manifolds as invalid.
    cell = tria.begin ();
    for (; cell!=endc; ++cell)
    {
      cell->set_all_manifold_ids(numbers::invalid_manifold_id);
    }
    // Now find those on the surface of the conductor.
    // Do this by looking through all cells with material_id
    // of the conductor, then finding any faces of those cells which
    // are shared with a cell with material_id of the non-conductor.
    cell = tria.begin ();
    for (; cell!=endc; ++cell)
    {
      if (cell->material_id() == 1)
      {
        for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
        {
          if (cell->neighbor(face)->material_id() == 0)
          {
            cell->face(face)->set_all_manifold_ids(100);
          }
        }
      }
    }
    
    // Here we set the boundary ids for the boundary conditions
    // and may choose to mark the boundary as curved, etc.
    // Can also perform mesh refinement here.
    
    // Set boundaries to neumann (boundary_id = 10)
    if (neumann_flag)
    {
      cell = tria.begin ();
      for (; cell!=endc; ++cell)
      {
        for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
        {
          if (cell->face(face)->at_boundary())
          {
            cell->face(face)->set_boundary_id (10);
          }
        }
      }
    }
  }
  template <int dim>
  void torusBenchmark<dim>::run(std::string input_filename, 
                                 std::string output_filename)
  {
    
    ParameterHandler prm;
    InputTools::ParameterReader param(prm);    
    param.read_and_copy_parameters(input_filename);
    
    InputTools::read_in_mesh<dim>(IO_Data::mesh_filename,
                                  tria);
    
    process_mesh(EquationData::neumann_flag);
   
    
    initialise_materials();    
        
    deallog << "Number of active cells:       "
    << tria.n_active_cells()
    << std::endl;
    
    // Now setup the forward problem:
    dof_handler.distribute_dofs (fe);
    
    const MappingQ<dim> mapping(mapping_order, (mapping_order>1 ? true : false));
    
    ForwardSolver::EddyCurrent<dim, DoFHandler<dim>> eddy(mapping,
                                                          dof_handler,
                                                          fe,
                                                          PreconditionerData::use_direct);
    
    deallog << "Number of degrees of freedom: "
    << dof_handler.n_dofs()
    << std::endl;
    
    // assemble the matrix for the eddy current problem:
    deallog << "Assembling System Matrix...." << std::endl;
    eddy.assemble_matrices(dof_handler);
    deallog << "Matrix Assembly complete. " << std::endl;
    
    // initialise the linear solver - precomputes any inverses for the preconditioner, etc:
    deallog << "Initialising Solver..." << std::endl;
    eddy.initialise_solver();
    deallog << "Solver initialisation complete. " << std::endl;
    
    // construct RHS for this field and
    // Take complex conjugate of polarization tensor:
    Vector<double> uniform_field(dim+dim);
    for (unsigned int d=0;d<dim;++d)
    {
      uniform_field(d) = PolarizationTensor::H0[0](d);
      uniform_field(d+dim) = PolarizationTensor::H0[1](d);
    }
    backgroundField::conductingObject_polarization_tensor<dim> 
      boundary_conditions(PolarizationTensor::H0,
                          PolarizationTensor::polarizationTensor);
    
    // assemble rhs
    deallog << "Assembling System RHS...." << std::endl;
    eddy.assemble_rhs(dof_handler,
                      boundary_conditions);
    deallog << "Matrix RHS complete. " << std::endl;
    
    deallog << "Solving... " << std::endl;
    Vector<double> solution;
    // solve system & storage in the vector of solutions:
    unsigned int n_gmres_iterations;
    eddy.solve(solution,n_gmres_iterations);

    deallog << "Computed solution. " << std::endl;
    
    // Display the l2 norm of the solution over the domain
    Vector<double> diff_per_cell(tria.n_active_cells());
    VectorTools::integrate_difference(mapping, dof_handler, solution, boundary_conditions,
                                      diff_per_cell, QGauss<dim>(2*(poly_order+1)+2), VectorTools::L2_norm);
    const double l2err = diff_per_cell.l2_norm();
    
    deallog << "L2 norm of solution: " << l2err << std::endl;
    // Short version:
    std::cout << tria.n_active_cells()
    << " " << dof_handler.n_dofs()
    << " " << n_gmres_iterations
    << " " << l2err
    << std::endl;
    
    {
      std::ostringstream tmp;
      tmp << output_filename;    
      OutputTools::output_to_vtk<dim, DoFHandler<dim>>(mapping,
                                                       dof_handler,
                                                       solution,
                                                       tmp.str(),
                                                       boundary_conditions);
    }
    
    // Output the solution fields along a line to a text file:
    Point <dim> xaxis;
    Point <dim> yaxis;
    Point <dim> zaxis;
    Point <dim> diagaxis;
    if (MeshData::boundary_shape=="cube")
    { 
      xaxis = Point<dim> (MeshData::xmax, 0.0, 0.0);
      yaxis = Point<dim> (0.0, MeshData::ymax, 0.0);
      zaxis = Point<dim> (0.0, 0.0, MeshData::zmax);
      diagaxis = Point<dim> (MeshData::xmax, MeshData::ymax, MeshData::zmax);
    }
    else if (MeshData::boundary_shape=="cylinder_z")
    {
      xaxis = Point<dim> (MeshData::radius, 0.0, 0.0);
      yaxis = Point<dim> (0.0, MeshData::radius, 0.0);
      zaxis = Point<dim> (0.0, 0.0, MeshData::zmax);
      diagaxis = Point<dim> (MeshData::radius*cos(numbers::PI/4.0), MeshData::radius*sin(numbers::PI/4.0), MeshData::zmax);
    }
    
    {
    std::stringstream tmp;
    tmp << output_filename << "_xaxis";
    std::string xaxis_str = tmp.str();
    OutputTools::output_radial_values<dim> (mapping,
                                            dof_handler,
                                            solution,
                                            boundary_conditions,
                                            uniform_field,
                                            xaxis,
                                            xaxis_str);
    }
    {
      std::stringstream tmp;
      tmp << output_filename << "_yaxis";
      std::string yaxis_str = tmp.str();
      OutputTools::output_radial_values<dim> (mapping,
                                              dof_handler,
                                              solution,
                                              boundary_conditions,
                                              uniform_field,
                                              yaxis,
                                              yaxis_str);
    }
    {
      std::stringstream tmp;
      tmp << output_filename << "_zaxis";
      std::string zaxis_str = tmp.str();
      OutputTools::output_radial_values<dim> (mapping,
                                              dof_handler,
                                              solution,
                                              boundary_conditions,
                                              uniform_field,
                                              zaxis,
                                              zaxis_str);
    }
    {
      std::stringstream tmp;
      tmp << output_filename << "_diagaxis";
      std::string diagaxis_str = tmp.str();
      OutputTools::output_radial_values<dim> (mapping,
                                              dof_handler,
                                              solution,
                                              boundary_conditions,
                                              uniform_field,
                                              diagaxis,
                                              diagaxis_str);
    }
  }
}

int main (int argc, char* argv[])
{
  using namespace dealii;
  
  const unsigned int dim = 3;
  // Set default input:
  unsigned int p_order = 0;
  unsigned int mapping_order = 1;
  std::string output_filename = "torus";
  std::string input_filename = "../input_files/torus_benchmark.prm";
  
  // Allow for input from command line:
  if (argc > 0)
  {
    for (int i=1;i<argc;i++)
    {
      if (i+1 != argc)
      {
        std::string input = argv[i];
        if (input == "-p")
        {
          std::stringstream strValue;
          strValue << argv[i+1];
          strValue >> p_order;
        }
        if (input == "-m")
        {
          std::stringstream strValue;
          strValue << argv[i+1];
          strValue >> mapping_order;
          if (mapping_order < 1)
          {
            std::cout << "ERROR: mapping order must be > 0" << std::endl;
            return 1;
          }
        }
        if (input == "-i")
        {
          input_filename = argv[i+1];
        }
        if (input == "-o")
        {
          output_filename = argv[i+1];
        }
      }
    }
  }
  
  // Only output to logfile, not console:
  deallog.depth_console(0);
  std::ostringstream deallog_filename;
  deallog_filename << output_filename << "_p" << p_order << ".deallog";
  std::ofstream deallog_file(deallog_filename.str());
  deallog.attach(deallog_file);
  
  torusBenchmark::torusBenchmark<dim> eddy_voltages(p_order,
                                                    mapping_order);
  eddy_voltages.run(input_filename,
                    output_filename);
  
  deallog_file.close();
  return 0;
}