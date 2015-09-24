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

#include <deal.II/fe/fe_nedelec.h>

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

namespace TEAMBenchmark7
{
  
  template <int dim>
  class TEAMBenchmark7
  {
  public:
    TEAMBenchmark7 (const unsigned int poly_order,
                     const unsigned int mapping_order = 2);
    ~TEAMBenchmark7 ();
    void run(std::string input_filename,
             std::string output_filename,
             const unsigned int href);
  private:
    Triangulation<dim> tria;
    FESystem<dim> fe;
    DoFHandler<dim> dof_handler;

    const unsigned int poly_order;
    const unsigned int mapping_order;
    
    void initialise_materials();
    void process_mesh(bool neuman_flag);
    
    void output_along_line(const Mapping<dim> &mapping,
                           const DoFHandler<dim> &dof_handler,
                           const Vector<double> &solution,
                           const EddyCurrentFunction<dim> &boundary_conditions,
                           const Point<dim> &start_point,
                           const Point<dim> &end_point,
                           const unsigned int n_points,
                           const std::string &filename);
  };
  
  template <int dim>
  TEAMBenchmark7<dim>::TEAMBenchmark7(const unsigned int poly_order,
                                        const unsigned int mapping_order)
  :
  fe (MyFE_Nedelec<dim>(poly_order), 2),
  dof_handler (tria),
  poly_order(poly_order),
  mapping_order(mapping_order)
  {
  }
  
  template <int dim>
  TEAMBenchmark7<dim>::~TEAMBenchmark7 ()
  {
    dof_handler.clear ();  
  }
  
  template <int dim>
  void TEAMBenchmark7<dim>::output_along_line(const Mapping<dim> &mapping,
                                              const DoFHandler<dim> &dof_handler,
                                              const Vector<double> &solution,
                                              const EddyCurrentFunction<dim> &boundary_conditions,
                                              const Point<dim> &start_point,
                                              const Point<dim> &end_point,
                                              const unsigned int n_points,
                                              const std::string &filename)
  {
    // Function to output computed and exact values
    // from the centre (0,0,0) towards the input end_point
    const unsigned int p_order = dof_handler.get_fe().degree-1;
    std::vector<Point<dim>> measurement_points(n_points);    
    
    // deliberately miss out boundary point as it's set by BCs
    for (unsigned int i=0; i<measurement_points.size(); ++i)
    {
      
      measurement_points[i] = start_point + (end_point-start_point)*(((double) i)/ ((double)(n_points-1)));
      
    }
    std::vector<Vector<double>> field_values_exact(measurement_points.size(),
                                                   Vector<double> (dim+dim));
    std::vector<Vector<double>> field_values_approx(measurement_points.size(),
                                                    Vector<double> (dim+dim));
    
    std::vector<Vector<double>> curl_values_exact(measurement_points.size(),
                                                  Vector<double> (dim+dim));
    std::vector<Vector<double>> curl_values_approx(measurement_points.size(),
                                                   Vector<double> (dim+dim));      
    
    boundary_conditions.vector_value_list(measurement_points,
                                          field_values_exact);                                      
    boundary_conditions.curl_value_list(measurement_points,
                                        curl_values_exact);
    
    for (unsigned int i=0; i<measurement_points.size(); ++i)
    {
      VectorTools::point_value(mapping,
                               dof_handler,
                               solution,
                               measurement_points[i],
                               field_values_approx[i]);
      
      Vector<double> temp_curl(dim+dim);
      MyVectorTools::point_curl(mapping,
                                dof_handler,
                                solution,
                                measurement_points[i],
                                temp_curl);
      for (unsigned int d=0; d<dim+dim; ++d)
      {
        curl_values_approx[i](d) = temp_curl(d);
      }
    }
    
    // output to file:
    std::ostringstream tmp;
    tmp << filename << "_fieldvals_p" << p_order << ".out";
    std::ofstream file(tmp.str());
    file.precision(32);
    for (unsigned int i=0; i<measurement_points.size(); ++i)
    {
      // 1 2 3
      file << measurement_points[i] << " "; 
      // approx field:
      // 4 5 6 7 8 9
      for (unsigned int d=0;d<dim+dim; ++d)
      {
        file << field_values_approx[i](d) << " ";
      }
      // exact field:
      // 10 11 12 13 14 15
      for (unsigned int d=0;d<dim+dim; ++d)
      {
        file << field_values_exact[i](d) << " ";
      }
      
      // approx curl
      // 16 17 18 19 20 21
      for (unsigned int d=0;d<dim+dim; ++d)
      {
        file << curl_values_approx[i](d) << " ";
      }
      // exact curl
      // 22 23 24 25 26 27
      for (unsigned int d=0;d<dim+dim; ++d)
      {
        file << curl_values_exact[i](d) << " ";
      }
      
      file << std::endl;
    }
    file.close();
  }
  
  template <int dim>
  void TEAMBenchmark7<dim>::initialise_materials()
  {
    EquationData::param_mur.reinit(3);
    EquationData::param_mur(0) = EquationData::param_mur_background;
    EquationData::param_mur(1) = EquationData::param_mur_conducting;
    EquationData::param_mur(2) = EquationData::param_mur_background;
    
    EquationData::param_sigma.reinit(3);
    EquationData::param_sigma(0) = EquationData::param_sigma_background;
    EquationData::param_sigma(1) = EquationData::param_sigma_conducting;
    EquationData::param_sigma(2) = EquationData::param_sigma_background;
    
    EquationData::param_epsilon.reinit(3);
    EquationData::param_epsilon(0) = EquationData::param_epsilon_background;
    EquationData::param_epsilon(1) = EquationData::param_epsilon_conducting;
    EquationData::param_epsilon(2) = EquationData::param_epsilon_background;
    
    
    // kappa = -omega^2*epr + i*omega*sigma;
    // i.e. kappa_re = -omega^2
    //      kappa_im = omega*sigma
    EquationData::param_kappa_re.reinit(EquationData::param_mur.size());
    EquationData::param_kappa_im.reinit(EquationData::param_mur.size());
    for (unsigned int i=0;i<EquationData::param_mur.size();i++) // note mur and kappa must have same size:
    {
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
      }
    }
    
    // We have a non-zero RHS, so must set the appropriate factor to multiply by
    EquationData::rhs_factor = EquationData::constant_mu0;
  }
  template <int dim>
  void TEAMBenchmark7<dim>::process_mesh(bool neumann_flag)
  {
    // Routine to process the read in mesh
    typename Triangulation<dim>::cell_iterator cell;
    const typename Triangulation<dim>::cell_iterator endc = tria.end();

    // Make the interior sphere's boundary a spherical boundary:    
    // First set all manifold_ids to 0 (default).
    cell = tria.begin ();
    for (; cell!=endc; ++cell)
    {
      cell->set_all_manifold_ids(numbers::invalid_manifold_id);
    }
    // Now find those on the surface of the sphere.
    // Do this by looking through all cells with material_id
    // of the conductor, then finding any faces of those cells which
    // are shared with a cell with material_id of the non-conductor.
    cell = tria.begin ();
    for (; cell!=endc; ++cell)
    {
      // First if cell lies on boundary of the inner sphere, then flag as spherical (manifold 100).
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
      // Also if cell lies on outer boundary, then flag as spherical (manifold 100).
      else if (cell->at_boundary())
      {
        for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
        {
          if (cell->face(face)->at_boundary())
          {
            cell->face(face)->set_all_manifold_ids(100);
          }
        }
      }
      if (cell->material_id() == 0)
      {
        cell-> set_all_manifold_ids(100);
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

    // Now refine the outer mesh
//     cell = tria.begin();
//     for (; cell!=endc; ++cell)
//     {
// //       if (cell->material_id() == 0)
// //       {
//         cell->set_refine_flag();
// //       }
//     }
//     tria.execute_coarsening_and_refinement ();
  }
  template <int dim>
  void TEAMBenchmark7<dim>::run(std::string input_filename,
                                 std::string output_filename,
                                 const unsigned int href)
  {
    ParameterHandler prm;
    InputTools::ParameterReader param(prm);    
    param.read_and_copy_parameters(input_filename);
    
    // Read in mesh (not option for internal mesh here).
    InputTools::read_in_mesh<dim>(IO_Data::mesh_filename,
                                  tria);
      
    process_mesh(EquationData::neumann_flag);
    if (href>0)
    {
      tria.refine_global(href);
    }
    
    initialise_materials();
    
    deallog << "Number of active cells:       "
    << tria.n_active_cells()
    << std::endl;
    
    // Now setup the forward problem:
    dof_handler.distribute_dofs (fe);
    
    // Mapping, higher order if required.
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
    
    // Create object for BCs and RHS.
    backgroundField::TEAMBenchmark7<dim> boundary_conditions(TEAMBenchmark::coil_centre,
                                                             TEAMBenchmark::corner_centres,
                                                             TEAMBenchmark::coil_material_id);
    
    
    // assemble rhs
    deallog << "Assembling System RHS...." << std::endl;
    eddy.assemble_rhs(dof_handler,
                      boundary_conditions);
    deallog << "Matrix RHS complete. " << std::endl;
    
    deallog << "Solving... " << std::endl;
    Vector<double> solution;
    // solve system & storage in the vector of solutions:
    unsigned int n_gmres_iterations;
    eddy.solve(solution, n_gmres_iterations);

    deallog << "Computed solution. " << std::endl;
    
    // Output error to screen:
    
    double l2err_wholeDomain;
    double l2err_conductor;
    double hcurlerr_wholeDomain;
    double hcurlerr_conductor;
    
    MyVectorTools::calcErrorMeasures(mapping,
                                     dof_handler,
                                     solution,
                                     boundary_conditions,
                                     l2err_wholeDomain,
                                     l2err_conductor,
                                     hcurlerr_wholeDomain,
                                     hcurlerr_conductor);
    
    deallog << "L2 Errors | Whole Domain: " << l2err_wholeDomain << "   Conductor Only:" << l2err_conductor << std::endl;
    deallog << "HCurl Error | Whole Domain: " << hcurlerr_wholeDomain << "   Conductor Only:" << hcurlerr_conductor << std::endl;
    // Short version:
    std::cout << tria.n_active_cells()
    << " " << dof_handler.n_dofs()
    << " " << n_gmres_iterations
    << " " << l2err_wholeDomain
    << " " << l2err_conductor
    << " " << hcurlerr_wholeDomain
    << " " << hcurlerr_conductor
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
    // Now output the values for the benchmark.
    const unsigned int n_benchmark_points = 4*17; // 4x the number required.
    std::vector<Point<dim>> start_point(4);
    std::vector<Point<dim>> end_point(4);
    std::vector<std::string> filename_str(4);
    start_point[0] = Point<dim> (0.0, 0.072, 0.034);
    end_point[0] = Point<dim> (0.288, 0.072, 0.034);
    filename_str[0] = output_filename + "_line1";
    start_point[1] = Point<dim> (0.0, 0.144, 0.034);
    end_point[1] = Point<dim> (0.288, 0.144, 0.034);
    filename_str[1] = output_filename + "_line2";
    start_point[2] = Point<dim> (0.0, 0.072, 0.019);
    end_point[2] = Point<dim> (0.288, 0.072, 0.019);
    filename_str[2] = output_filename + "_line3";
    start_point[3] = Point<dim> (0.0, 0.072, 0.0);
    end_point[3] = Point<dim> (0.288, 0.072, 0.0);
    filename_str[3] = output_filename + "_line4";
    for (unsigned int i=0; i<4;++i)
    {
      output_along_line(mapping,
                        dof_handler,
                        solution,
                        boundary_conditions,
                        start_point[i],
                        end_point[i],
                        n_benchmark_points,
                        filename_str[i]);
    }
  }
}

int main (int argc, char* argv[])
{
  using namespace dealii;
  
  const int dim = 3;
  // Set default input:
  unsigned int p_order = 0;
  unsigned int href = 0;
  unsigned int mapping_order = 1;
  std::string output_filename = "teambenchmark7";
  std::string input_filename = "../input_files/teambenchmark7.prm";
  
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
        if (input == "-h") // h refinement
        {
          std::stringstream strValue;
          strValue << argv[i+1];
          strValue >> href;
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
  
  TEAMBenchmark7::TEAMBenchmark7<dim> eddy_voltages(p_order,
                                                    mapping_order);
  eddy_voltages.run(input_filename,
                    output_filename,
                    href);
  
  deallog_file.close();
  return 0;
}