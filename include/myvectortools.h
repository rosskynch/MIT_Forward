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

#ifndef MYVECTORTOOLS_H
#define MYVECTORTOOLS_H

// deal.II includes:
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/hp/dof_handler.h>

#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/vector_tools.h>

// std includes:

// My includes:
//#include <backgroundfield.h>
#include <curlfunction.h>

using namespace dealii;

namespace MyVectorTools
{
  // Calculates the Hcurl error for the input solution against
  // the input EddyCurrentFunction<dim> exact solution.
  //
  // NOTE: the solution must correspond to a real/imaginary part of an
  //       FE_Nedelec finite element.
  template<int dim, class DH>
  double calcErrorHcurlNorm(const DH &dof_handler,
                            const Vector<double> &solution,
                            const EddyCurrentFunction<dim> &exact_solution);
  template<int dim, class DH>
  double calcErrorHcurlNorm(const Mapping<dim> &mapping,
                            const DH &dof_handler,
                            const Vector<double> &solution,
                            const EddyCurrentFunction<dim> &exact_solution);
  
  template<int dim, class DH>
  void calcErrorMeasures(const DH &dof_handler,
                         const Vector<double> &solution,
                         const EddyCurrentFunction<dim> &exact_solution,
                         double &l2err_all,
                         double &l2err_conductor,
                         double &hcurlerr_all,
                         double &hcurlerr_conductor);
  template<int dim, class DH>
  void calcErrorMeasures(const Mapping<dim> &mapping,
                         const DH &dof_handler,
                         const Vector<double> &solution,
                         const EddyCurrentFunction<dim> &exact_solution,
                         double &l2err_all,
                         double &l2err_conductor,
                         double &hcurlerr_all,
                         double &hcurlerr_conductor);
  
  // Functions to return the gradient values of an FE solution at a point:
  // Note: only works for FE_Nedelec elements with 2 vector valued blocks
  //       which correspond to the real & imaginary part.
  template <int dim, class DH>
  void point_gradient(const DH &dof_handler,
                      const Vector<double> &solution,
                      const Point<dim> &point,
                      std::vector<Tensor<1,dim>> &gradients);
  template <int dim, class DH>
  void point_gradient(const Mapping<dim> &mapping,
                      const DH &dof_handler,
                      const Vector<double> &solution,
                      const Point<dim> &point,
                      std::vector<Tensor<1,dim>> &gradients);
  
  
  // Functions to return the curls of an FE solution at a point:
  // Note: only works for FE_Nedelec elements with 2 vector valued blocks
  //       which correspond to the real & imaginary part.
  template <int dim, class DH>
  void point_curl (const DH &dof_handler,
                   const Vector<double> &solution,
                   const Point<dim> &point,
                   Vector<double> &curl);
  template <int dim, class DH>
  void point_curl (const Mapping<dim> &mapping,
                   const DH &dof_handler,
                   const Vector<double> &solution,
                   const Point<dim> &point,
                   Vector<double> &curl);
}
#endif