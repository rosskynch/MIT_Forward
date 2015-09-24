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

#ifndef CURLFUNCTION_H
#define CURLFUNCTION_H

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

#include <deal.II/lac/vector.h>


using namespace dealii;

// TODO: We should be able to remove the curlfunction and perturbedfunction classes.
//       They're now completely replacable by the EddyCurrentFunction class, which is far more comprehensive.
//       This will need some changes to many of the other classes/functions, so will leave alone for now.


template<int dim>
class curlFunction : public Function<dim>
{
public:
//   curlFunction ();//unsigned int n_components = dim+dim);
  virtual void curl_value_list (const std::vector<Point<dim> > &points,
                                std::vector<Vector<double> >   &values) const = 0;
                                
  //virtual void perturbed_field_value_list (const std::vector<Point<dim> > &points,
  //                                         std::vector<Vector<double> >   &values) const = 0;
};

template<int dim>
class perturbedFunction : public curlFunction<dim>
{
public:
//   perturbedFunction ();//unsigned int n_components = dim+dim);
  virtual void curl_value_list (const std::vector<Point<dim> > &points,
                                std::vector<Vector<double> >   &values) const = 0;
                                
  virtual void perturbed_field_value_list (const std::vector<Point<dim> > &points,
                                           std::vector<Vector<double> >   &values) const = 0;
};

// Base Function class for all EddyCurrent problems.
// The idea here is to allow implementation of boundary conditions of the type:
// n x E = f (Dirichlet) and n x curl(E) = g (Neumann)
//
// which can be handled through the vector_value_list and curl_value_list functions.
//
// The RHS of the equation can be handled through the rhs_value_list.
//
// The perturbed_field_value_list can be used for comparison with analytical/asymptotic formulae.
//
// We also include the boolean function zero_xxx() for each to return if the function is simply zero.
// this allows us to avoid wasted computation which are automatically zero (e.g. in RHS assembly).
//
// We also include the option of passing a material_id to the function. Since we are typically
// dealing with subdomains where we have conducting/non-conducting objects, this allows communication
// from structure in the mesh to the function, if required (default is off).
template<int dim>
class EddyCurrentFunction : public Function<dim>
{
public:
  // Leave out constructor for now, we may need to create one which
  // calls the Function constructor with n_components = dim+dim by default).
  //EddyCurrentFunction (unsigned int n_components = dim+dim);
  
  // We overload each member so that it implements the style found in Function<dim>
  // as well as allowing the use of the material_id if required.
  // The obvious way to do this is to implement the Function<dim>-style version by
  // calling the material_id version with numbers::invalid_material_id,
  // effectively making it the default value for that argument.
  //
  // Obviously derived classes an bypass this if required.
  
  virtual void vector_value_list (const std::vector<Point<dim> > &points,
                                  std::vector<Vector<double> >   &values) const;
  
  virtual void vector_value_list (const std::vector<Point<dim> > &points,
                                  std::vector<Vector<double> >   &values,
                                  const types::material_id &mat_id) const;
                                  
  virtual void curl_value_list (const std::vector<Point<dim> > &points,
                                std::vector<Vector<double> >   &values) const;

  virtual void curl_value_list (const std::vector<Point<dim> > &points,
                                std::vector<Vector<double> >   &values,
                                const types::material_id &mat_id) const;
                                
  virtual void perturbed_field_value_list (const std::vector<Point<dim> > &points,
                                           std::vector<Vector<double> >   &values) const;

  virtual void perturbed_field_value_list (const std::vector<Point<dim> > &points,
                                           std::vector<Vector<double> >   &values,
                                           const types::material_id &mat_id) const;

  virtual void rhs_value_list (const std::vector<Point<dim> > &points,
                               std::vector<Vector<double> >   &values) const;
                               
  virtual void rhs_value_list (const std::vector<Point<dim> > &points,
                               std::vector<Vector<double> >   &values,
                               const types::material_id &mat_id) const;

  virtual bool zero_vector() const { return true; }
  virtual bool zero_curl() const { return true; }
  virtual bool zero_perturbed() const { return true; }
  virtual bool zero_rhs() const { return true; }
};
#endif