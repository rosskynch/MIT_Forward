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

#include <myvectortools.h>

using namespace dealii;

namespace MyVectorTools
{
  template<int dim, class DH>
  double calcErrorHcurlNorm(const DH &dof_handler,
                            const Vector<double> &solution,
                            const EddyCurrentFunction<dim> &exact_solution)
  {
    // Default Q1 mapping, calls the full version below.
    return calcErrorHcurlNorm (StaticMappingQ1<dim>::mapping,
                               dof_handler,
                               solution,
                               exact_solution);
  }
  template<int dim, class DH>
  double calcErrorHcurlNorm(const Mapping<dim> &mapping,
                            const DH &dof_handler,
                            const Vector<double> &solution,
                            const EddyCurrentFunction<dim> &exact_solution)
  { 
    const FiniteElement<dim> &fe = dof_handler.get_fe ();
    
    const unsigned int quad_order = 2*fe.degree + 1;
    
    QGauss<dim>  quadrature_formula(quad_order);
    const unsigned int n_q_points = quadrature_formula.size();
    
    FEValues<dim> fe_values (mapping, fe, quadrature_formula,
                             update_values    |  update_gradients |
                             update_quadrature_points  |  update_JxW_values);
    
    // Extractors to real and imaginary parts
    const FEValuesExtractors::Vector E_re(0);
    const FEValuesExtractors::Vector E_im(dim);
    
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
    
    // storage for exact sol:
    std::vector<Vector<double> > exactsol(n_q_points, Vector<double>(fe.n_components()));
    std::vector<Vector<double> > exactcurlsol(n_q_points, Vector<double>(fe.n_components()));
    Tensor<1,dim>  errSol_re;
    Tensor<1,dim>  errSol_im;
    Tensor<1,dim>  errCurlSol_re;
    Tensor<1,dim>  errCurlSol_im;
    
    // storage for computed sol:
    std::vector<Tensor<1,dim> > sol_re(n_q_points);
    std::vector<Tensor<1,dim> > sol_im(n_q_points);
    std::vector<Tensor<1,dim> > curlsol_re(n_q_points);
    std::vector<Tensor<1,dim> > curlsol_im(n_q_points);
    
    double h_curl_norm=0.0;
    
    typename DH::active_cell_iterator cell = dof_handler.begin_active();
    const typename DH::active_cell_iterator endc = dof_handler.end();
    for (; cell!=endc; ++cell)
    {
      fe_values.reinit (cell);
      
      // Store exact values of E and curlE:
      exact_solution.vector_value_list(fe_values.get_quadrature_points(), exactsol);
      exact_solution.curl_value_list(fe_values.get_quadrature_points(), exactcurlsol);
      
      // Store computed values at quad points:
      fe_values[E_re].get_function_values(solution, sol_re);
      fe_values[E_im].get_function_values(solution, sol_im);
      fe_values[E_re].get_function_curls(solution, curlsol_re);
      fe_values[E_im].get_function_curls(solution, curlsol_im);
      
      // Calc values of curlE from fe solution:
      cell->get_dof_indices (local_dof_indices);
      // Loop over quad points to calculate solution:
      for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
      {
        // Split exact solution into real/imaginary parts:
        for (unsigned int d=0; d<dim; d++)
        {
          errSol_re[d] = exactsol[q_point][d] - sol_re[q_point][d];
          errSol_im[d] = exactsol[q_point][d+dim] - sol_im[q_point][d];
          errCurlSol_re[d] = exactcurlsol[q_point][d] - curlsol_re[q_point][d];
          errCurlSol_im[d] = exactcurlsol[q_point][d+dim] - curlsol_im[q_point][d];
        }
        // Integrate difference at each point:             
        h_curl_norm += ( errSol_re*errSol_re 
                        + errSol_im*errSol_im
                        + errCurlSol_re*errCurlSol_re
                        + errCurlSol_im*errCurlSol_im
                       )*fe_values.JxW(q_point);
      }
    }
    return sqrt(h_curl_norm);
  }
  // Template instantiation:
  template double calcErrorHcurlNorm<>(const DoFHandler<3> &,
                                       const Vector<double> &,
                                       const EddyCurrentFunction<3> &);
  template double calcErrorHcurlNorm<>(const Mapping<3> &,
                                       const DoFHandler<3> &,
                                       const Vector<double> &,
                                       const EddyCurrentFunction<3> &);
  
  template<int dim, class DH>
  void calcErrorMeasures(const DH &dof_handler,
                         const Vector<double> &solution,
                         const EddyCurrentFunction<dim> &exact_solution,
                         double &l2err_all,
                         double &l2err_conductor,
                         double &hcurlerr_all,
                         double &hcurlerr_conductor)
  {
    // Default Q1 mapping, calls the full version below.
    calcErrorMeasures (StaticMappingQ1<dim>::mapping,
                       dof_handler,
                       solution,
                       exact_solution,
                       l2err_all,
                       l2err_conductor,
                       hcurlerr_all,
                       hcurlerr_conductor);
  }
  template<int dim, class DH>
  void calcErrorMeasures(const Mapping<dim> &mapping,
                         const DH &dof_handler,
                         const Vector<double> &solution,
                         const EddyCurrentFunction<dim> &exact_solution,
                         double &l2err_all,
                         double &l2err_conductor,
                         double &hcurlerr_all,
                         double &hcurlerr_conductor)
  { 
    const FiniteElement<dim> &fe = dof_handler.get_fe ();
    
    const unsigned int quad_order = 2*fe.degree + 1;
    
    QGauss<dim>  quadrature_formula(quad_order);
    const unsigned int n_q_points = quadrature_formula.size();
    
    FEValues<dim> fe_values (mapping, fe, quadrature_formula,
                             update_values    |  update_gradients |
                             update_quadrature_points  |  update_JxW_values);
    
    // Extractors to real and imaginary parts
    const FEValuesExtractors::Vector E_re(0);
    const FEValuesExtractors::Vector E_im(dim);
    
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
    
    // storage for exact sol:
    std::vector<Vector<double> > exactsol(n_q_points, Vector<double>(fe.n_components()));
    std::vector<Vector<double> > exactcurlsol(n_q_points, Vector<double>(fe.n_components()));
    Tensor<1,dim>  errSol_re;
    Tensor<1,dim>  errSol_im;
    Tensor<1,dim>  errCurlSol_re;
    Tensor<1,dim>  errCurlSol_im;
    
    // storage for computed sol:
    std::vector<Tensor<1,dim> > sol_re(n_q_points);
    std::vector<Tensor<1,dim> > sol_im(n_q_points);
    std::vector<Tensor<1,dim> > curlsol_re(n_q_points);
    std::vector<Tensor<1,dim> > curlsol_im(n_q_points);
    
    double soltemp_all = 0;
    double curltemp_all = 0;
    double soltemp_conductor = 0;
    double curltemp_conductor = 0;
    
    typename DH::active_cell_iterator cell = dof_handler.begin_active();
    const typename DH::active_cell_iterator endc = dof_handler.end();
    for (; cell!=endc; ++cell)
    {
      fe_values.reinit (cell);

      // Store exact values of E and curlE:
      exact_solution.vector_value_list(fe_values.get_quadrature_points(), exactsol);
      exact_solution.curl_value_list(fe_values.get_quadrature_points(), exactcurlsol);
      
      // Store computed values at quad points:
      fe_values[E_re].get_function_values(solution, sol_re);
      fe_values[E_im].get_function_values(solution, sol_im);
      fe_values[E_re].get_function_curls(solution, curlsol_re);
      fe_values[E_im].get_function_curls(solution, curlsol_im);
      
      // Calc values of curlE from fe solution:
      cell->get_dof_indices (local_dof_indices);
      // Loop over quad points to calculate solution:
      double curl_part = 0;
      double sol_part = 0;
      for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
      {
        // Split exact solution into real/imaginary parts:
        for (unsigned int d=0; d<dim; d++)
        {
          errSol_re[d] = exactsol[q_point][d] - sol_re[q_point][d];
          errSol_im[d] = exactsol[q_point][d+dim] - sol_im[q_point][d];
          errCurlSol_re[d] = exactcurlsol[q_point][d] - curlsol_re[q_point][d];
          errCurlSol_im[d] = exactcurlsol[q_point][d+dim] - curlsol_im[q_point][d];
        }
        // Integrate difference at each point:
        
        sol_part += (errSol_re*errSol_re + errSol_im*errSol_im)*fe_values.JxW(q_point);
        curl_part += (errCurlSol_re*errCurlSol_re + errCurlSol_im*errCurlSol_im)*fe_values.JxW(q_point);
      }
      if (cell->material_id() == 1)
      {
        soltemp_conductor += sol_part;
        curltemp_conductor += curl_part;
      }
      soltemp_all += sol_part;
      curltemp_all += curl_part;
    }
    l2err_all = sqrt(soltemp_all);
    l2err_conductor = sqrt(soltemp_conductor);
    hcurlerr_all = sqrt(soltemp_all + curltemp_all);
    hcurlerr_conductor = sqrt(soltemp_conductor + curltemp_conductor);
  }
  // Template instantiation:
  template void calcErrorMeasures<>(const DoFHandler<3> &,
                                    const Vector<double> &,
                                    const EddyCurrentFunction<3> &,
                                    double &,
                                    double &,
                                    double &,
                                    double &);
  template void calcErrorMeasures<>(const Mapping<3> &,
                                    const DoFHandler<3> &,
                                    const Vector<double> &,
                                    const EddyCurrentFunction<3> &,
                                    double &,
                                    double &,
                                    double &,
                                    double &);

  // Functions to return the derived values of a FE solution at a point:
  template <int dim, class DH>
  void point_gradient(const DH &dof_handler,
                      const Vector<double> &solution,
                      const Point<dim> &point,
                      std::vector<Tensor<1,dim>> &gradients)
  {
    // Default Q1 mapping, calls the full version below.
    point_gradient (StaticMappingQ1<dim>::mapping,
                    dof_handler,
                    solution,
                    point,
                    gradients);
  }
  
  template <int dim, class DH>
  void point_gradient(const Mapping<dim> &mapping,
                      const DH &dof_handler,
                      const Vector<double> &solution,
                      const Point<dim> &point,
                      std::vector<Tensor<1,dim>> &gradients)
  {
    // Compute the gradients of the a vector-valued finite element
    // solution at an arbitrary point.
    // gradients[c][d] with contain the derivative in coord direct d of the
    // c-th vector component of the vector field.
    
    const std::pair<typename DH::active_cell_iterator, Point<dim> >
      cell_point = GridTools::find_active_cell_around_point(mapping, dof_handler, point);
    
    const double delta = 1e-10;
    Assert(GeometryInfo<dim>::distance_to_unit_cell(cell_point.second) < delta,
           ExcInternalError());
    
    const Quadrature<dim>
    quadrature (GeometryInfo<dim>::project_to_unit_cell(cell_point.second));
      
    FEValues<dim> fe_values(mapping, dof_handler.get_fe(), quadrature,
                            update_values | update_gradients | update_quadrature_points );
    fe_values.reinit(cell_point.first);
    
    const FEValuesExtractors::Vector v_re (0);
    const FEValuesExtractors::Vector v_im (dim);
    std::vector<Tensor<2,dim>>
      temp_gradient_re(1);
    std::vector<Tensor<2,dim>>
      temp_gradient_im(1);
      
    fe_values[v_re].get_function_gradients(solution, temp_gradient_re);
    fe_values[v_im].get_function_gradients(solution, temp_gradient_im);
    for (unsigned int i=0; i<dim; ++i)
    {
      for (unsigned int d=0; d<dim; ++d)
      {
        gradients[i][d] = temp_gradient_re[0][i][d];
        gradients[i+dim][d] = temp_gradient_im[0][i][d];
      }
    }
  }
  // Template instantiation:
  template void point_gradient<>(const DoFHandler<3> &,
                                 const Vector<double> &,
                                 const Point<3> &,
                                 std::vector<Tensor<1,3>> &);
  template void point_gradient<>(const Mapping<3> &,
                                 const DoFHandler<3> &,
                                 const Vector<double> &,
                                 const Point<3> &,
                                 std::vector<Tensor<1,3>> &);
  
  template <int dim, class DH>
  void point_curl (const DH &dof_handler,
                   const Vector<double> &solution,
                   const Point<dim> &point,
                   Vector<double> &curl)
  {
    // Default Q1 mapping, calls the full version below.
    point_curl (StaticMappingQ1<dim>::mapping,
                dof_handler,
                solution,
                point,
                curl);
  }
  
  template <int dim, class DH>
  void point_curl (const Mapping<dim> &mapping,
                   const DH &dof_handler,
                   const Vector<double> &solution,
                   const Point<dim> &point,
                   Vector<double> &curl)
  {
    // Compute the curl of a vector-valued finite element
    // solution at an arbitrary point.
    // curl[c][d] will return the d-th component of the curl of the c-th
    // component of the vector field.
    // This is only valid for 3-D at the moment.
    
    
    const unsigned int dofs_per_cell = dof_handler.get_fe().dofs_per_cell;
    
    const std::pair<typename DH::active_cell_iterator, Point<dim> >
      cell_point = GridTools::find_active_cell_around_point(mapping, dof_handler, point);
    
    const double delta = 1e-6;
    Assert(GeometryInfo<dim>::distance_to_unit_cell(cell_point.second) < delta,
           ExcInternalError());
    
    const Quadrature<dim>
    quadrature (GeometryInfo<dim>::project_to_unit_cell(cell_point.second));
      
    FEValues<dim> fe_values(mapping, dof_handler.get_fe(), quadrature,
                            update_values | update_gradients |
                            update_quadrature_points);
    fe_values.reinit(cell_point.first);
    
    const FEValuesExtractors::Vector v_re (0);
    const FEValuesExtractors::Vector v_im (dim);
    std::vector< Tensor<1,dim> >
      temp_curl_re(1);
    std::vector< Tensor<1,dim> >
      temp_curl_im(1);
      
    // Get curls:
    fe_values[v_re].get_function_curls(solution, temp_curl_re);
    fe_values[v_im].get_function_curls(solution, temp_curl_im);
    /*
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
    cell_point.first->get_dof_indices (local_dof_indices);
    Tensor<1,dim> curl_re;
    Tensor<1,dim> curl_im;
    curl_re=0.0;
    curl_im=0.0;
    for (unsigned int i=0; i<dofs_per_cell; ++i)
    {
      // gradient to curl. let g be grad:
      // curl[0] = g[2][1] - g[1][2]
      // curl[0] = g[0][2] - g[2][0]
      // curl[0] = g[1][0] - g[0][1]
      unsigned int block_index_i = dof_handler.get_fe().system_to_block_index(i).first;
      if (block_index_i==0)
      {
//         curl_re += solution(local_dof_indices[i])*fe_values[v_re].curl(i,0);
        // grad way:
        curl_re[0] += solution(local_dof_indices[i])*(fe_values[v_re].gradient(i,0)[2][1]
                                                     -fe_values[v_re].gradient(i,0)[1][2]);
        
        curl_re[1] += solution(local_dof_indices[i])*(fe_values[v_re].gradient(i,0)[0][2]
                                                     -fe_values[v_re].gradient(i,0)[2][0]);
        
        curl_re[2] += solution(local_dof_indices[i])*(fe_values[v_re].gradient(i,0)[1][0]
                                                     -fe_values[v_re].gradient(i,0)[0][1]);
                                                      
        
      }
      else if (block_index_i == 1)
      {
//         curl_im +=solution(local_dof_indices[i])*fe_values[v_im].curl(i,0);
                // grad way:
        curl_im[0] += solution(local_dof_indices[i])*(fe_values[v_im].gradient(i,0)[2][1]
                                                     -fe_values[v_im].gradient(i,0)[1][2]);
        
        curl_im[1] += solution(local_dof_indices[i])*(fe_values[v_im].gradient(i,0)[0][2]
                                                     -fe_values[v_im].gradient(i,0)[2][0]);
        
        curl_im[2] += solution(local_dof_indices[i])*(fe_values[v_im].gradient(i,0)[1][0]
                                                     -fe_values[v_im].gradient(i,0)[0][1]);
      }
      
    }
    */
         
    for (unsigned int i=0; i<dim; ++i)
    {
      curl(i)=temp_curl_re[0][i];
      curl(i+dim)=temp_curl_im[0][i];
/*
      curl(i)=curl_re[i];
      curl(i+dim)=curl_im[i];
*/
    }    
  }
  // Template instantiation:
  template void point_curl<> (const DoFHandler<3> &,
                              const Vector<double> &,
                              const Point<3> &,
                              Vector<double> &);
  template void point_curl<> (const Mapping<3> &,
                              const DoFHandler<3> &,
                              const Vector<double> &,
                              const Point<3> &,
                              Vector<double> &);

}