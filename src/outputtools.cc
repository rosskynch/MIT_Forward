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

#include <outputtools.h>

using namespace dealii;

namespace OutputTools
{
  // Postprocessor functions:
  template <int dim>
  Postprocessor<dim>::Postprocessor (const EddyCurrentFunction<dim> &exact_solution)
  :
  DataPostprocessor<dim>(),
  exact_solution(&exact_solution)
  {    
  }
  
  template <int dim>
  std::vector<std::string>
  Postprocessor<dim>::get_names() const
  {
    
    std::vector<std::string> solution_names (dim, "sol_re");
    
    solution_names.push_back ("sol_im");
    solution_names.push_back ("sol_im");
    solution_names.push_back ("sol_im");
    
    solution_names.push_back ("curlsol_re");
    solution_names.push_back ("curlsol_re");
    solution_names.push_back ("curlsol_re");
    
    solution_names.push_back ("curlsol_im");
    solution_names.push_back ("curlsol_im");
    solution_names.push_back ("curlsol_im");
    
    // Exact solution:
    solution_names.push_back ("exact_re");
    solution_names.push_back ("exact_re");
    solution_names.push_back ("exact_re");
    
    solution_names.push_back ("exact_im");
    solution_names.push_back ("exact_im");
    solution_names.push_back ("exact_im");
    
    // Exact solution:
    solution_names.push_back ("exactcurl_re");
    solution_names.push_back ("exactcurl_re");
    solution_names.push_back ("exactcurl_re");
    
    solution_names.push_back ("exactcurl_im");
    solution_names.push_back ("exactcurl_im");
    solution_names.push_back ("exactcurl_im");
    
    // Exact solution:
    solution_names.push_back ("error_re");
    solution_names.push_back ("error_re");
    solution_names.push_back ("error_re");
    
    solution_names.push_back ("error_im");
    solution_names.push_back ("error_im");
    solution_names.push_back ("error_im");
    
    // Exact solution:
    solution_names.push_back ("errorcurl_re");
    solution_names.push_back ("errorcurl_re");
    solution_names.push_back ("errorcurl_re");
    
    solution_names.push_back ("errorcurl_im");
    solution_names.push_back ("errorcurl_im");
    solution_names.push_back ("errorcurl_im");
    
    return solution_names;
  }
  template <int dim>
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  Postprocessor<dim>::
  get_data_component_interpretation () const
  {
    // for solution re/imag:
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    interpretation (dim,
                    DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    
    // for curl of solution re/imag:
    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    

    // For exact solution:
    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    
    // For exact curl solution (Alternatively, perturbed curl solution):
    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    
    // For errors in solution
    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    
    // For errors in curl of solution
    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    
    return interpretation;
  }
  template <int dim>
  UpdateFlags
  Postprocessor<dim>::get_needed_update_flags() const
  {
    return update_values | update_gradients | update_hessians | update_q_points; //update_normal_vectors |
  }
  template <int dim>
  void
  Postprocessor<dim>::compute_derived_quantities_vector (const std::vector<Vector<double> >    &uh,
                                                         const std::vector<std::vector<Tensor<1,dim> > >      &duh,
                                                         const std::vector<std::vector<Tensor<2,dim> > >      &dduh,
                                                         const std::vector<Point<dim> >                       &normals,
                                                         const std::vector<Point<dim> >                       &evaluation_points,
                                                         std::vector<Vector<double> >                         &computed_quantities) const
  {
    const unsigned int n_quadrature_points = uh.size();
    Assert (duh.size() == n_quadrature_points,                  ExcInternalError());
    Assert (computed_quantities.size() == n_quadrature_points,  ExcInternalError());
    Assert (uh[0].size() == dim+dim,                            ExcInternalError());
    
    std::vector<Vector<double> > solution_value_list(n_quadrature_points, Vector<double>(dim+dim));
    std::vector<Vector<double> > h_value_list(n_quadrature_points, Vector<double>(dim+dim));
    exact_solution->vector_value_list(evaluation_points, solution_value_list);
    exact_solution->curl_value_list(evaluation_points, h_value_list);
    
    for (unsigned int q=0; q<n_quadrature_points; ++q)
    {
      // Electric field, E:
      for (unsigned int d=0; d<dim + dim; ++d)
      {
        computed_quantities[q](d) = uh[q](d);
      }
      
      // CurlE:
      // TODO: Check this is correct.
      // If we have A, is E=-i*omega*A. H = mu*E
      // real&imaginary part of curl:
      for (unsigned int c=0; c<2; ++c)
      {
        // duh[q][c][d] < q'th quad point, c'th vector (of length dim), d'th co-ordinate.
        // 0, 1, 2 = real part, 3, 4, 5 = imaginary part, so c*dim traverses.
        computed_quantities[q](0 + c*dim + 2*dim) = (duh[q][2+c*dim][1]-duh[q][1+c*dim][2]);
        computed_quantities[q](1 + c*dim + 2*dim) = (duh[q][0+c*dim][2]-duh[q][2+c*dim][0]);
        computed_quantities[q](2 + c*dim + 2*dim) = (duh[q][1+c*dim][0]-duh[q][0+c*dim][1]);        
      }
//       computed_quantities[q](0+2*dim) = (duh[q][2][1]-duh[q][1][2]);
//       computed_quantities[q](1+2*dim) = (duh[q][0][2]-duh[q][2][0]);
//       computed_quantities[q](2+2*dim) = (duh[q][1][0]-duh[q][0][1]);
//       //imaginary part
//       computed_quantities[q](0+3*dim) = (duh[q][5][1]-duh[q][4][2]);
//       computed_quantities[q](1+3*dim) = (duh[q][3][2]-duh[q][5][0]);
//       computed_quantities[q](2+3*dim) = (duh[q][4][0]-duh[q][3][1]);
      
      // Exact Solution & Errors:
      for (unsigned int i=0; i<dim+dim; i++)
      {
        // exact soln:
        computed_quantities[q](i+4*dim) = solution_value_list[q](i); // uh[q](i) (if wanting to compute error).
        // exact curl:
        computed_quantities[q](i+6*dim) = h_value_list[q](i);
        
        // soln error:
        computed_quantities[q](i+8*dim) = solution_value_list[q](i) - uh[q](i);
        // curl error
        computed_quantities[q](i+10*dim) = h_value_list[q](i) - computed_quantities[q](i+2*dim); ;
      }
    }
  }
  template class Postprocessor<3>;
  
  // Outputs to VTK:
    template<int dim, class DH>
  void output_to_vtk (const DH &dof_handler,
                      const Vector<double> &solution,
                      const std::string &vtk_filename,
                      const EddyCurrentFunction<dim> &exact_solution)
  {
    // Default Q1 mapping, calls the full version below.
    output_to_vtk(StaticMappingQ1<dim>::mapping,
                  dof_handler,
                  solution,
                  vtk_filename,
                  exact_solution);
  }
  template<int dim, class DH>
  void output_to_vtk (const Mapping<dim> &mapping,
                      const DH &dof_handler,
                      const Vector<double> &solution,
                      const std::string &vtk_filename,
                      const EddyCurrentFunction<dim> &exact_solution)
  {
    unsigned int n_subdivisions = IO_Data::n_subdivisions;
    unsigned int p_order = dof_handler.get_fe().degree-1;
    
    std::ostringstream filename;
    filename << vtk_filename << "_p" << p_order << ".vtk";
    std::ofstream output (filename.str().c_str());
    
    // postprocessor handles all quantities to output
    // NOTE IT MUST GO BEFORE DataOut<dim>!!!
    OutputTools::Postprocessor<dim> postprocessor(exact_solution);
    
    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);
    
    data_out.add_data_vector (solution, postprocessor);
    
    data_out.build_patches (mapping, n_subdivisions, DataOut<dim>::curved_inner_cells);
    data_out.write_vtk (output);
    
    // Section to append the material parameters onto the end of the VTK file:
    // First part taken from how deal.ii outputs a scalar:
    output << "SCALARS sigma double 1" << std::endl;
    output << "LOOKUP_TABLE default" << std::endl;
    // now loop over all active cells:
    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    unsigned int n_points_per_cell;
    if (n_subdivisions > 0)
    {
      n_points_per_cell=pow(n_subdivisions+1,3);
    }
    else
    {
      n_points_per_cell=8;
    }
    for (; cell!=endc; ++cell)
    {
      // need to output enough to cover all the quad points, which is based on
      // the number of subdivisions
      for (unsigned int i=0; i<n_points_per_cell; ++i)
      {
        output << EquationData::param_sigma(cell->material_id()) << " ";
      }
    }
    // Add mur
    output << std::endl;
    output << "SCALARS mur double 1" << std::endl;
    output << "LOOKUP_TABLE default" << std::endl;
    cell=dof_handler.begin_active();
    for (; cell!=endc; ++cell)
    {
      // need to output enough to cover all the quad points, which is based on
      // the number of subdivisions
      for (unsigned int i=0; i<n_points_per_cell; ++i)
      {
        output << EquationData::param_mur(cell->material_id()) << " ";
      }
    }
    
    // Add omega
    output << std::endl;
    output << "SCALARS omega double 1" << std::endl;
    output << "LOOKUP_TABLE default" << std::endl;
    cell=dof_handler.begin_active();
    for (; cell!=endc; ++cell)
    {
      // need to output enough to cover all the quad points, which is based on
      // the number of subdivisions
      for (unsigned int i=0; i<n_points_per_cell; ++i)
      {
        output << EquationData::param_omega << " ";
      }
    }
    output.close();
  }
  // Template instantiation:
  template void output_to_vtk<>(const DoFHandler<3> &,
                                const Vector<double> &,
                                const std::string &,
                                const EddyCurrentFunction<3> &);
  template void output_to_vtk<>(const Mapping<3> &,
                                const DoFHandler<3> &,
                                const Vector<double> &,
                                const std::string &,
                                const EddyCurrentFunction<3> &);
  
  template <int dim, class DH>
  void output_radial_values(const DH &dof_handler,
                            const Vector<double> &solution,
                            const EddyCurrentFunction<dim> &boundary_conditions,
                            const Vector<double> &uniform_field,
                            const Point<dim> &end_point,
                            const std::string &filename)
  {
    // Default Q1 mapping, calls the full version below.
    output_radial_values(StaticMappingQ1<dim>::mapping,
                         dof_handler,
                         solution,
                         boundary_conditions,
                         uniform_field,
                         end_point,
                         filename);
  }
  
  template <int dim, class DH>
  void output_radial_values(const Mapping<dim> &mapping,
                            const DH &dof_handler,
                            const Vector<double> &solution,
                            const EddyCurrentFunction<dim> &boundary_conditions,
                            const Vector<double> &uniform_field,
                            const Point<dim> &end_point,
                            const std::string &filename)
  {
    // Function to output computed and exact values
    // from the centre (0,0,0) towards the input end_point
    unsigned int p_order = dof_handler.get_fe().degree-1;
    unsigned int n_points = 100;
    std::vector<Point<dim>> measurement_points(n_points);

    const double inc = 1.0/n_points;
    
    // deliberately miss out boundary point as it's set by BCs
    for (unsigned int i=0; i<measurement_points.size(); ++i)
    {
      measurement_points[i] = i*inc*end_point;
      
    }
    std::vector<Vector<double>> field_values_exact(measurement_points.size(),
                                                   Vector<double> (dim+dim));
    std::vector<Vector<double>> field_values_approx(measurement_points.size(),
                                                    Vector<double> (dim+dim));
    
    std::vector<Vector<double>> curl_values_exact(measurement_points.size(),
                                                  Vector<double> (dim+dim));
    std::vector<Vector<double>> curl_values_approx(measurement_points.size(),
                                                   Vector<double> (dim+dim));      
    
    std::vector<Vector<double>> perturbed_field_values_exact(measurement_points.size(),
                                                             Vector<double> (dim+dim));
    std::vector<Vector<double>> perturbed_field_values_approx(measurement_points.size(),
                                                              Vector<double> (dim+dim));
    
    boundary_conditions.vector_value_list(measurement_points,
                                          field_values_exact);                                      
    boundary_conditions.curl_value_list(measurement_points,
                                        curl_values_exact);
    boundary_conditions.perturbed_field_value_list(measurement_points,
                                                   perturbed_field_values_exact);
    
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
        perturbed_field_values_approx[i](d) = temp_curl(d) - uniform_field(d);
      }
    }
    
    // output to file:
    std::ostringstream tmp;
    tmp << filename << "_ptfield_p" << p_order << ".out";
    std::ofstream file(tmp.str());
    file.precision(32);
    for (unsigned int i=0; i<measurement_points.size(); ++i)
    {
      double r = sqrt(measurement_points[i].square());
      file << r << " ";
      // approx field:
      for (unsigned int d=0;d<dim+dim; ++d)
      {
        file << field_values_approx[i](d) << " ";
      }
      // exact field:
      for (unsigned int d=0;d<dim+dim; ++d)
      {
        file << field_values_exact[i](d) << " ";
      }
      
      // approx curl
      for (unsigned int d=0;d<dim+dim; ++d)
      {
        file << curl_values_approx[i](d) << " ";
      }
      // exact curl
      for (unsigned int d=0;d<dim+dim; ++d)
      {
        file << curl_values_exact[i](d) << " ";
      }
      
      // approx curl perturbed
      for (unsigned int d=0;d<dim+dim; ++d)
      {
        file << perturbed_field_values_approx[i](d) << " ";
      }
      // exact curl perturbed
      for (unsigned int d=0;d<dim+dim; ++d)
      {
        file << perturbed_field_values_exact[i](d) << " ";
      }        
      
      file << std::endl;
    }
    file.close();
  }
  // Template instantiation:
  template void output_radial_values(const DoFHandler<3> &,
                                     const Vector<double> &,
                                     const EddyCurrentFunction<3> &,
                                     const Vector<double> &,
                                     const Point<3> &,                            
                                     const std::string &);
  template void output_radial_values(const Mapping<3> &,
                                     const DoFHandler<3> &,
                                     const Vector<double> &,
                                     const EddyCurrentFunction<3> &,
                                     const Vector<double> &,
                                     const Point<3> &,                            
                                     const std::string &);

  
}