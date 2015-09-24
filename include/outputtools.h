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

#ifndef OUTPUTTOOLS_H
#define OUTPUTTOOLS_H

// deal.II includes:
#include <deal.II/base/data_out_base.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/hp/dof_handler.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_postprocessor.h>

// std includes:
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

// My includes:
#include <all_data.h>
#include <curlfunction.h>
#include <myvectortools.h>

using namespace dealii;

namespace OutputTools
{
  // Postprocessor for use by output_to_vtk
  template <int dim>
  class Postprocessor : public DataPostprocessor<dim>
  {
  public:
    // TODO: add functionality to pass a function class to this.
    Postprocessor (const EddyCurrentFunction<dim> &exact_solution);
    virtual
    void
    compute_derived_quantities_vector (const std::vector<Vector<double> >              &uh,
                                       const std::vector<std::vector<Tensor<1,dim> > > &duh,
                                       const std::vector<std::vector<Tensor<2,dim> > > &dduh,
                                       const std::vector<Point<dim> >                  &normals,
                                       const std::vector<Point<dim> >                  &evaluation_points,
                                       std::vector<Vector<double> >                    &computed_quantities) const;
                                       
    virtual std::vector<std::string> get_names () const;
    virtual
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    get_data_component_interpretation () const;
    virtual UpdateFlags get_needed_update_flags () const;
    
  private:
    const SmartPointer< const EddyCurrentFunction<dim> > exact_solution;
    
  };
  
  template<int dim, class DH>
  void output_to_vtk (const DH &dof_handler,
                      const Vector<double> &solution,
                      const std::string &vtk_filename,
                      const EddyCurrentFunction<dim> &exact_solution);
  template<int dim, class DH>
  void output_to_vtk (const Mapping<dim> &mapping,
                      const DH &dof_handler,
                      const Vector<double> &solution,
                      const std::string &vtk_filename,
                      const EddyCurrentFunction<dim> &exact_solution);
  
  // TODO ADD output_radial_perturbed_values.
  // Then remove the perturbed parts from the output_radial_values routines.
  /*
  template <int dim, class DH>
  void output_radial_perturbed_values(const DH &dof_handler,
                            const Vector<double> &solution,
                            const EddyCurrentFunction<dim> &boundary_conditions,
                            const Vector<double> &uniform_field,
                            const Point<dim> &end_point,
                            const std::string &filename);*/
  
  template <int dim, class DH>
  void output_radial_values(const DH &dof_handler,
                            const Vector<double> &solution,
                            const EddyCurrentFunction<dim> &boundary_conditions,
                            const Vector<double> &uniform_field,
                            const Point<dim> &end_point,
                            const std::string &filename);
  template <int dim, class DH>
  void output_radial_values(const Mapping<dim> &mapping,
                            const DH &dof_handler,
                            const Vector<double> &solution,
                            const EddyCurrentFunction<dim> &boundary_conditions,
                            const Vector<double> &uniform_field,
                            const Point<dim> &end_point,
                            const std::string &filename);
}
#endif