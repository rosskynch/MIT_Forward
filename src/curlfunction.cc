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

#include <curlfunction.h>

using namespace dealii;

/*
template<int dim>
class curlFunction : public Function<dim>
{
public:
  virtual void curl_value_list (const std::vector<Point<dim> > &points,
                                std::vector<Vector<double> >   &values) const = 0;
  
  //virtual void perturbed_field_value_list (const std::vector<Point<dim> > &points,
  //                                         std::vector<Vector<double> >   &values) const = 0;
};
*/
// template<int dim>
// curlFunction<dim>::curlFunction()//unsigned int n_components)
// :
// // Function<dim> (n_components)
// {
// }
template class curlFunction<3>;
/*
template<int dim>
class perturbedFunction : public curlFunction<dim>
{
public:
  virtual void curl_value_list (const std::vector<Point<dim> > &points,
                                std::vector<Vector<double> >   &values) const = 0;
                                
  virtual void perturbed_field_value_list (const std::vector<Point<dim> > &points,
                                           std::vector<Vector<double> >   &values) const = 0;
};
*/
// template<int dim>
// perturbedFunction<dim> ::perturbedFunction()//unsigned int n_components)
// :
// // curlFunction<dim> (n_components)
// {}
template class perturbedFunction<3>;


// Implementation of the virtual functions for EddyCurrentFunction:
// In all cases we just return zero, and so set the zero_xxx boolean function to return true.
// This means that any derived classes will need to implement pairs of the appropriate xxx_value_list
// and zero_xxx (returning false) functions.
template<int dim>
void EddyCurrentFunction<dim>::vector_value_list (const std::vector<Point<dim> > &points,
                                                     std::vector<Vector<double> >   &values) const
{
  vector_value_list(points, values, numbers::invalid_material_id);
}

template<int dim>
void EddyCurrentFunction<dim>::vector_value_list (const std::vector<Point<dim> > &points,
                                                     std::vector<Vector<double> >   &values,
                                                     const types::material_id &mat_id) const
{
  Assert(values.size() == points.size(), ExcDimensionMismatch(values.size(), points.size()));
  for (unsigned int k=0; k<points.size(); ++k)
  {
    values[k]=0;
  }
}

template<int dim>
void EddyCurrentFunction<dim>::curl_value_list (const std::vector<Point<dim> > &points,
                                                     std::vector<Vector<double> >   &values) const
{
  curl_value_list(points, values, numbers::invalid_material_id);
}

template<int dim>
void EddyCurrentFunction<dim>::curl_value_list (const std::vector<Point<dim> > &points,
                                                   std::vector<Vector<double> >   &values,
                                                   const types::material_id &mat_id) const
{
  Assert(values.size() == points.size(), ExcDimensionMismatch(values.size(), points.size()));
  for (unsigned int k=0; k<points.size(); ++k)
  {
    values[k]=0;
  }
}

template<int dim>
void EddyCurrentFunction<dim>::perturbed_field_value_list (const std::vector<Point<dim> > &points,
                                                     std::vector<Vector<double> >   &values) const
{
  perturbed_field_value_list(points, values, numbers::invalid_material_id);
}

template<int dim>
void EddyCurrentFunction<dim>::perturbed_field_value_list (const std::vector<Point<dim> > &points,
                                                              std::vector<Vector<double> >   &values,
                                                              const types::material_id &mat_id) const
{
  Assert(values.size() == points.size(), ExcDimensionMismatch(values.size(), points.size()));
  for (unsigned int k=0; k<points.size(); ++k)
  {
    values[k]=0;
  }
}

template<int dim>
void EddyCurrentFunction<dim>::rhs_value_list (const std::vector<Point<dim> > &points,
                                                     std::vector<Vector<double> >   &values) const
{
  rhs_value_list(points, values, numbers::invalid_material_id);
}

template<int dim>
void EddyCurrentFunction<dim>::rhs_value_list (const std::vector<Point<dim> > &points,
                                                  std::vector<Vector<double> >   &values,
                                                  const types::material_id &mat_id) const
{
  Assert(values.size() == points.size(), ExcDimensionMismatch(values.size(), points.size()));
  for (unsigned int k=0; k<points.size(); ++k)
  {
    values[k]=0;
  }
}

template class EddyCurrentFunction<3>;