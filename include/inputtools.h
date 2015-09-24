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

#ifndef INPUTTOOLS_H
#define INPUTTOOLS_H

// deal.II includes:
#include <deal.II/base/parameter_handler.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

// std includes:
#include <fstream>
#include <iostream>
#include <algorithm>

// My includes:
#include <all_data.h>

using namespace dealii;

namespace InputTools
{
  template <int dim>
  void read_in_mesh (const std::string mesh_name,
                     Triangulation<dim> &triangulation);
  
  class ParameterReader : public Subscriptor
  {
  public:
    ParameterReader(ParameterHandler &paramhandler);
    ~ParameterReader();
    void read_parameters (const std::string);
    void read_and_copy_parameters (const std::string);
    void copy_parameters ();
    
  private:
    void declare_parameters();
    void copy_to_equation_data();
    void get_matrix_from_list(const std::string entry,
                              FullMatrix<double> &matrix_out,
                              const unsigned int mRows,
                              const unsigned int nCols);
    
    void get_vector_from_list(const std::string entry,
                              Vector<double> &vector_out,
                              const unsigned int vector_length = 0);
    
    unsigned int get_vector_length_from_list(const std::string entry,
                                             const char delimiter);
    ParameterHandler &prm;
  };
  
}
#endif
