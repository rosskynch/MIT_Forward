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

#include <all_data.h>
#include <backgroundfield.h>
#include <curlfunction.h>
#include <forwardsolver.h>
#include <inputtools.h>
#include <myvectortools.h>
#include <outputtools.h>

#include <myfe_nedelec.h>

using namespace dealii;

namespace polynomialBenchmark
{
  
  template <int dim>
  class polynomialBenchmark
  {
  public:
    polynomialBenchmark (const unsigned int order);
    ~polynomialBenchmark ();
    void run(std::string input_filename,
             std::string output_filename);
  private:
    Triangulation<dim> tria;
    FESystem<dim> fe;
    DoFHandler<dim> dof_handler;

    unsigned int p_order;
    
    void initialise_materials();
    void process_mesh(bool neuman_flag);
  };
  
  template <int dim>
  polynomialBenchmark<dim>::polynomialBenchmark(const unsigned int order)
  :
  tria (Triangulation<3>::MeshSmoothing::none, true),
  fe (MyFE_Nedelec<dim>(order), 2),
  dof_handler (tria),
  p_order(order)
  {
  }
  
  template <int dim>
  polynomialBenchmark<dim>::~polynomialBenchmark ()
  {
    dof_handler.clear ();  
  }
  
  template <int dim>
  void polynomialBenchmark<dim>::initialise_materials()
  {
    EquationData::rhs_factor = 1.0;
    // Fill the equation data, but ultimately override it anyway.
    EquationData::param_mur.reinit(2);
    EquationData::param_mur(0) = 1.0;
    EquationData::param_mur(1) = 1.0;
    
    EquationData::param_sigma.reinit(2);
    EquationData::param_sigma(0) = 0.0;
    EquationData::param_sigma(1) = 0.0;
    
    EquationData::param_epsilon.reinit(2);
    EquationData::param_epsilon(0) = 0.0;
    EquationData::param_epsilon(1) = 0.0;
    
    // We're solving curl(curl(A)) + A = J, mur = 1 and kappa = 1 + 0i.
    EquationData::param_kappa_re.reinit(2);
    EquationData::param_kappa_im.reinit(2);
    EquationData::param_kappa_re(0) = 1.0;
    EquationData::param_kappa_re(1) = 1.0;
    EquationData::param_kappa_im(0) = 0.0;
    EquationData::param_kappa_im(1) = 0.0;

  }
  template <int dim>
  void polynomialBenchmark<dim>::process_mesh(bool neumann_flag)
  {
    // Routine to process the mesh for this problem
    // Here we set the boundary ids for the boundary conditions
    // and may choose to mark the boundary as curved, etc.
    // Can also perform mesh refinement here.
    // Also need to add material_ids.
    
    typename Triangulation<dim>::cell_iterator
    cell = tria.begin (),
    endc = tria.end ();
    
    // Set material id to 1 (all one material, as set in intialise_materials(), above).
    // We choose 1 as this is the default for a conductor in the eddy current solver
    // and we don't want any regularisation.
    for (; cell!=endc; ++cell)
    {
      cell->set_material_id(1);
    }
   
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
            cell->face(face)->set_all_boundary_ids (10);
          }
        }
      }
    }
    else
    {
      cell = tria.begin ();
      for (; cell!=endc; ++cell)
      {
        for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
        {
          if (cell->face(face)->at_boundary())
          {
            cell->face(face)->set_all_boundary_ids (0);
          }
        }
      }
    }
  }
  template <int dim>
  void polynomialBenchmark<dim>::run(std::string input_filename, 
                                 std::string output_filename)
  {
    
    ParameterHandler prm;
    InputTools::ParameterReader param(prm);
    param.read_and_copy_parameters(input_filename);
    
    // Allow for choice of mesh completely by input file:
    if (MeshData::external_mesh)
    {
      InputTools::read_in_mesh<dim>(MeshData::mesh_filename,
                                    tria);
//       tria.refine_global(2);
      if (MeshData::boundary_shape == "cube_distorted")
      {
        GridTools::distort_random (0.2, tria, false);  
      }
    }
    else
    {
      if (MeshData::boundary_shape == "cube" || MeshData::boundary_shape == "cube_distorted")
      {
        GridGenerator::hyper_cube (tria,
                                   MeshData::xmin,
                                   MeshData::xmax);
        tria.refine_global (1);
        if (MeshData::boundary_shape == "cube_distorted")
        {
          GridTools::distort_random (0.2, tria, false);
        }
      }
      else if (MeshData::boundary_shape == "cylinder")
      {
        GridGenerator::cylinder (tria, MeshData::radius, MeshData::height);
        static const CylinderBoundary<dim> cyl_boundary (MeshData::radius);
        tria.set_boundary (0, cyl_boundary);
//         tria.refine_global(1);
        GridTools::distort_random (0.2, tria, true);
      }
      else if (MeshData::boundary_shape == "sphere")
      {
        GridGenerator::hyper_ball (tria, Point<dim> (0.0,0.0,0.0), MeshData::radius);
        static const HyperBallBoundary<dim> sph_boundary (Point<dim> (0.0,0.0,0.0), MeshData::radius);
        tria.set_boundary (0, sph_boundary);
        tria.refine_global(1);
      }
    }
    process_mesh(EquationData::neumann_flag);
    
    // Set boundary condition. This also doubles as the RHS function (J = A).
    backgroundField::polynomialTest<dim> boundary_conditions;
    
    // Override any other value of omega here
    // as it must be such that omega = |k|
    EquationData::param_omega = 1.0;
    
    
    initialise_materials();
    
    
    std::cout << "Number of active cells:       "
    << tria.n_active_cells()
    << std::endl;
    

    
    // Now setup the forward problem:
    dof_handler.distribute_dofs (fe);
    ForwardSolver::EddyCurrent<dim, DoFHandler<dim>> eddy(dof_handler,
                                                          fe,
                                                          PreconditionerData::use_direct);
    
    std::cout << "Number of degrees of freedom: "
    << dof_handler.n_dofs()
    << std::endl;
    
    // assemble the matrix for the eddy current problem:
    std::cout << "Assembling System Matrix...." << std::endl;
    eddy.assemble_matrices(dof_handler);
    std::cout << "Matrix Assembly complete. " << std::endl;
    
    // initialise the linear solver - precomputes any inverses for the preconditioner, etc:
    std::cout << "Initialising Solver..." << std::endl;
    eddy.initialise_solver();
    std::cout << "Solver initialisation complete. " << std::endl;
    
    // Now solve for each excitation coil:
    std::cout << "Assembling RHS..." << std::endl;
    Vector<double> solution;
    
    // assemble rhs (using the non-zero rhs version).
    eddy.assemble_rhs(dof_handler,
                      boundary_conditions);
    
    std::cout << "RHS Assembly complete. " << std::endl;
    
    // solve system & storage in the vector of solutions:
    std::cout << "Solving... " << std::endl;
    unsigned int n_iterations;
    eddy.solve(solution, n_iterations);

    std::cout << "Computed solution. " << std::endl;
    {
      std::ostringstream tmp;
      tmp << output_filename;    
      OutputTools::output_to_vtk<dim, DoFHandler<dim>>(dof_handler,
                                                       solution,
                                                       tmp.str(),
                                                       boundary_conditions);
    }
    
    // Output error to screen:
    Vector<double> diff_per_cell(tria.n_active_cells());
    VectorTools::integrate_difference(dof_handler, solution, boundary_conditions,
                                      diff_per_cell, QGauss<dim>(2*p_order+3), VectorTools::L2_norm);
    const double l2err = diff_per_cell.l2_norm();
    double hcurlerr = MyVectorTools::calcErrorHcurlNorm(dof_handler,
                                                        solution,
                                                        boundary_conditions);
    std::cout << "L2 Error: " << l2err << " | HCurl Error: " << hcurlerr << std::endl;
  }
}

int main (int argc, char* argv[])
{
//  using namespace dealii;
  
  const int dim = 3;
  // Set default input:
  unsigned int p_order = 0;
  std::string output_filename = "poly_bench";
  std::string input_filename = "../input_files/poly_benchmark.prm";
  
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
  /* REMOVED:
  std::ostringstream deallog_filename;
  deallog_filename << output_filename << "_p" << p_order << ".deallog";
  std::ofstream deallog_file(deallog_filename.str());
  deallog.attach(deallog_file);
  */ 
  
  polynomialBenchmark::polynomialBenchmark<dim> poly_test(p_order);
  poly_test.run(input_filename,
                output_filename);
  
//   deallog_file.close();
  return 0;
}
