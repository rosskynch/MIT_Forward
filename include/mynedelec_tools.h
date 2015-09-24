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

#ifndef MYNEDELECTOOLS_H
#define MYNEDELECTOOLS_H

#include <deal.II/base/derivative_form.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/parallel_vector.h>
#include <deal.II/lac/parallel_block_vector.h>
// #include <deal.II/lac/petsc_vector.h>
// #include <deal.II/lac/petsc_block_vector.h>
// #include <deal.II/lac/trilinos_vector.h>
// #include <deal.II/lac/trilinos_block_vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/vector_memory.h>
#include <deal.II/lac/filtered_matrix.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_boundary.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/intergrid_map.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/hp/dof_handler.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>


#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/base/std_cxx11/array.h>
#include <numeric>
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
#include <set>
#include <list>

#include <map>
#include <vector>
#include <set>
#include <myfe_nedelec.h>

namespace MyNedelecTools
{
  namespace internals
  {
    template<typename cell_iterator>
    void
    compute_edge_projection_l2 (const cell_iterator &cell,
                                const unsigned int face,
                                const unsigned int line,
                                hp::FEValues<3> &hp_fe_values,
                                const Function<3> &boundary_function,
                                const unsigned int first_vector_component,
                                std::vector<double> &dof_values,
                                std::vector<bool> &dofs_processed)
    {
      // This function computes the L2-projection of the given
      // boundary function on 3D edges and returns the constraints
      // associated with the edge functions for the given cell.
      //
      // In the context of this function, by associated DoFs we mean:
      // the DoFs corresponding to the group of components making up the vector
      // with first component first_vector_component (length dim).
      const unsigned int dim = 3;
      const FiniteElement<dim> &fe = cell->get_fe ();

      // reinit for this cell, face and line.
      hp_fe_values.reinit
      (cell,
       (cell->active_fe_index () * GeometryInfo<dim>::faces_per_cell + face)
       * GeometryInfo<dim>::lines_per_face + line);

      // Initialize the required objects.
      const FEValues<dim> &
      fe_values = hp_fe_values.get_present_fe_values ();
      // Store degree as fe.degree-1
      // For nedelec elements FE_Nedelec<dim> (0) returns fe.degree = 1.
      const unsigned int degree = fe.degree - 1;

      const std::vector<Point<dim> > &
      quadrature_points = fe_values.get_quadrature_points ();
      std::vector<Vector<double> > values (fe_values.n_quadrature_points,
                                           Vector<double> (fe.n_components ()));

      // Get boundary function values
      // at quadrature points.
      boundary_function.vector_value_list (quadrature_points, values);

      // Find the group of vector components (dim of them,
      // starting at first_vector_component) are within an FESystem.
      //
      // If not using FESystem then must be using FE_Nedelec,
      // which has one base element and one copy of it (with 3 components).
      std::pair<unsigned int, unsigned int> base_indices (0, 0);
      if (dynamic_cast<const FESystem<dim>*> (&cell->get_fe ()) != 0)
        {
          unsigned int fe_index = 0;
          unsigned int fe_index_old = 0;
          unsigned int i = 0;

          // Find base element:
          // base_indices.first
          //
          // Then select which copy of that base element
          // [ each copy is of length fe.base_element(base_indices.first).n_components() ]
          // corresponds to first_vector_component:
          // base_index.second
          for (; i < fe.n_base_elements (); ++i)
            {
              fe_index_old = fe_index;
              fe_index += fe.element_multiplicity (i) * fe.base_element (i).n_components ();

              if (fe_index > first_vector_component)
                break;
            }

          base_indices.first = i;
          base_indices.second = (first_vector_component - fe_index_old) / fe.base_element (i).n_components ();
        }

      // Find DoFs we want to constrain:
      // There are fe.dofs_per_line DoFs associated with the
      // given line on the given face on the given cell.
      //
      // Want to know which of these DoFs (there are degree+1 of interest)
      // are associated with the components given by first_vector_component.
      // Then we can make a map from the associated line DoFs to the face DoFs.
      //
      // For a single FE_Nedelec<3> element this is simple:
      //    We know the ordering of local DoFs goes
      //    lines -> faces -> cells
      //
      // For a set of FESystem<3> elements we need to pick out the matching base element and
      // the index within this ordering.
      //
      // We call the map associated_edge_dof_to_face_dof
      std::vector<unsigned int> associated_edge_dof_to_face_dof (degree + 1);

      // Lowest DoF in the base element allowed for this edge:
      const unsigned int lower_bound =
        fe.base_element(base_indices.first).face_to_cell_index(line * (degree + 1), face);
      // Highest DoF in the base element allowed for this edge:
      const unsigned int upper_bound =
        fe.base_element(base_indices.first).face_to_cell_index((line + 1) * (degree + 1) - 1, face);

      unsigned int associated_edge_dof_index = 0;
      //       for (unsigned int face_idx = 0; face_idx < fe.dofs_per_face; ++face_idx)
      for (unsigned int line_idx = 0; line_idx < fe.dofs_per_line; ++line_idx)
        {
          // Assuming DoFs on a face are numbered in order by lines then faces.
          // i.e. line 0 has degree+1 dofs numbered 0,..,degree
          //      line 1 has degree+1 dofs numbered (degree+1),..,2*(degree+1) and so on.
          const unsigned int face_idx = line*fe.dofs_per_line + line_idx;
          // Note, assuming that the edge orientations are "standard"
          //       i.e. cell->line_orientation(line) = true.
          const unsigned int cell_idx = fe.face_to_cell_index (face_idx, face);

          // Check this cell_idx belongs to the correct base_element, component and line:
          if (((dynamic_cast<const FESystem<dim>*> (&fe) != 0)
               && (fe.system_to_base_index (cell_idx).first == base_indices)
               && (lower_bound <= fe.system_to_base_index (cell_idx).second)
               && (fe.system_to_base_index (cell_idx).second <= upper_bound))
              || ( (
                  (dynamic_cast<const FE_Nedelec<dim>*> (&fe) != 0)
                  || (dynamic_cast<const MyFE_Nedelec<dim>*> (&fe) != 0) )
                  && (line * (degree + 1) <= face_idx)
                  && (face_idx <= (line + 1) * (degree + 1) - 1)))
            {
              associated_edge_dof_to_face_dof[associated_edge_dof_index] = face_idx;
              ++associated_edge_dof_index;
            }
        }
      // Sanity check:
      const unsigned int associated_edge_dofs = associated_edge_dof_index;
      Assert (associated_edge_dofs == degree + 1,
              ExcMessage ("Error: Unexpected number of 3D edge DoFs"));

      // Matrix and RHS vectors to store linear system:
      // We have (degree+1) basis functions for an edge
      FullMatrix<double> edge_matrix (degree + 1,degree + 1);
      FullMatrix<double> edge_matrix_inv (degree + 1,degree + 1);
      Vector<double> edge_rhs (degree + 1);
      Vector<double> edge_solution (degree + 1);

      const FEValuesExtractors::Vector vec (first_vector_component);

      // coordinate directions of
      // the edges of the face.
      const unsigned int cell_line_index 
      = GeometryInfo< dim >::face_to_cell_lines(face, line,
                                                cell->face_orientation(face),
                                                cell->face_flip(face),
                                                cell->face_rotation(face));
      const unsigned int vertex0 (GeometryInfo<dim>::line_to_cell_vertices(cell_line_index,0));
      const unsigned int vertex1 (GeometryInfo<dim>::line_to_cell_vertices(cell_line_index,1));
      Point<dim> p0(cell->vertex(vertex0));
      Point<dim> p1(cell->vertex(vertex1));
      Tensor<1,dim> tangential = p1-p0;
      tangential /= tangential.norm();
      
//       const unsigned int
//       edge_coordinate_direction
//       [GeometryInfo<dim>::faces_per_cell]
//       [GeometryInfo<dim>::lines_per_face]
//       = { { 2, 2, 1, 1 },
//         { 2, 2, 1, 1 },
//         { 0, 0, 2, 2 },
//         { 0, 0, 2, 2 },
//         { 1, 1, 0, 0 },
//         { 1, 1, 0, 0 }
//       };
// 
//       const double tol = 0.5 * cell->face (face)->line (line)->diameter () / fe.degree;
//       const std::vector<Point<dim> > &
//       reference_quadrature_points = fe_values.get_quadrature ().get_points ();

      // Project the boundary function onto the shape functions for this edge
      // and set up a linear system of equations to get the values for the DoFs
      // associated with this edge.
//       std::cout << "CELL: " << cell << std::endl;
      for (unsigned int q_point = 0; q_point < fe_values.n_quadrature_points; ++q_point)
        {
          // Compute the tangential
          // of the edge at
          // the quadrature point.
//           Point<dim> shifted_reference_point_1 = reference_quadrature_points[q_point];
//           Point<dim> shifted_reference_point_2 = reference_quadrature_points[q_point];
// 
//           shifted_reference_point_1 (edge_coordinate_direction[face][line]) += tol;
//           shifted_reference_point_2 (edge_coordinate_direction[face][line]) -= tol;
//           Tensor<1,dim> tangential
//             = (0.5 *
//                (fe_values.get_mapping ()
//                 .transform_unit_to_real_cell (cell,
//                                               shifted_reference_point_1)
//                 -
//                 fe_values.get_mapping ()
//                 .transform_unit_to_real_cell (cell,
//                                               shifted_reference_point_2))
//                / tol);
//           tangential
//           /= tangential.norm ();
          
//           std::cout << q_point << " | "
//           << reference_quadrature_points[q_point] << " | "
//           << fe_values.get_mapping ().transform_unit_to_real_cell (cell, reference_quadrature_points[q_point])
//           << " | " << tangential << std::endl;

          // Compute the entires of the linear system
          // Note the system is symmetric so we could only compute the lower/upper triangle.
          //
          // The matrix entries are
          // \int_{edge} (tangential*edge_shape_function_i)*(tangential*edge_shape_function_j) dS
          //
          // The RHS entries are:
          // \int_{edge} (tangential*boundary_value)*(tangential*edge_shape_function_i) dS.
          for (unsigned int j = 0; j < associated_edge_dofs; ++j)
            {
              const unsigned int j_face_idx = associated_edge_dof_to_face_dof[j];
              const unsigned int j_cell_idx = fe.face_to_cell_index (j_face_idx, face);
              for (unsigned int i = 0; i < associated_edge_dofs; ++i)
                {
                  const unsigned int i_face_idx = associated_edge_dof_to_face_dof[i];
                  const unsigned int i_cell_idx = fe.face_to_cell_index (i_face_idx, face);

                  edge_matrix(i,j)
                  += fe_values.JxW (q_point)
                     * (fe_values[vec].value (i_cell_idx, q_point) * tangential)
                     * (fe_values[vec].value (j_cell_idx, q_point) * tangential);
                }
              // Compute the RHS entries:
              edge_rhs(j) += fe_values.JxW (q_point)
                             * (values[q_point] (first_vector_component) * tangential [0]
                                + values[q_point] (first_vector_component + 1) * tangential [1]
                                + values[q_point] (first_vector_component + 2) * tangential [2])
                             * (fe_values[vec].value (j_cell_idx, q_point) * tangential);
            }
        }

      // Invert linear system
      edge_matrix_inv.invert(edge_matrix);
      edge_matrix_inv.vmult(edge_solution,edge_rhs);

      // Store computed DoFs
      for (unsigned int i = 0; i < associated_edge_dofs; ++i)
        {
          dof_values[associated_edge_dof_to_face_dof[i]] = edge_solution(i);
          dofs_processed[associated_edge_dof_to_face_dof[i]] = true;
        }
    }


    template<int dim, typename cell_iterator>
    void
    compute_edge_projection_l2 (const cell_iterator &,
                                const unsigned int,
                                const unsigned int,
                                hp::FEValues<dim> &,
                                const Function<dim> &,
                                const unsigned int,
                                std::vector<double> &,
                                std::vector<bool> &)
    {
      // dummy implementation of above function
      // for all other dimensions
      Assert (false, ExcInternalError ());
    }

    template<int dim, typename cell_iterator>
    void
    compute_face_projection_curl_conforming_l2 (const cell_iterator &cell,
                                                const unsigned int face,
                                                hp::FEFaceValues<dim> &hp_fe_face_values,
                                                const Function<dim> &boundary_function,
                                                const unsigned int first_vector_component,
                                                std::vector<double> &dof_values,
                                                std::vector<bool> &dofs_processed)
    {
      // This function computes the L2-projection of the boundary
      // function on the interior of faces only. In 3D, this should only be
      // called after first calling compute_edge_projection_l2, as it relies on
      // edge constraints which are found.

      // In the context of this function, by associated DoFs we mean:
      // the DoFs corresponding to the group of components making up the vector
      // with first component first_vector_component (with total components dim).

      // Copy to the standard FEFaceValues object:
      hp_fe_face_values.reinit (cell, face);
      const FEFaceValues<dim> &fe_face_values = hp_fe_face_values.get_present_fe_values();

      // Initialize the required objects.
      const FiniteElement<dim> &fe = cell->get_fe ();
      const std::vector<Point<dim> > &
      quadrature_points = fe_face_values.get_quadrature_points ();
      const unsigned int degree = fe.degree - 1;

      std::vector<Vector<double> >
      values (fe_face_values.n_quadrature_points, Vector<double> (fe.n_components ()));

      // Get boundary function values at quadrature points.
      boundary_function.vector_value_list (quadrature_points, values);

      // Find where the group of vector components (dim of them,
      // starting at first_vector_component) are within an FESystem.
      //
      // If not using FESystem then must be using FE_Nedelec,
      // which has one base element and one copy of it (with 3 components).
      std::pair<unsigned int, unsigned int> base_indices (0, 0);
      if (dynamic_cast<const FESystem<dim>*> (&cell->get_fe ()) != 0)
        {
          unsigned int fe_index = 0;
          unsigned int fe_index_old = 0;
          unsigned int i = 0;

          // Find base element:
          // base_indices.first
          //
          // Then select which copy of that base element
          // [ each copy is of length fe.base_element(base_indices.first).n_components() ]
          // corresponds to first_vector_component:
          // base_index.second
          for (; i < fe.n_base_elements (); ++i)
            {
              fe_index_old = fe_index;
              fe_index += fe.element_multiplicity (i) * fe.base_element (i).n_components ();

              if (fe_index > first_vector_component)
                break;
            }
          base_indices.first = i;
          base_indices.second = (first_vector_component - fe_index_old) / fe.base_element (i).n_components ();
        }

      switch (dim)
        {
        case 2:
          // NOTE: This is very similar to compute_edge_projection as used in 3D,
          //       and contains a lot of overlap with that function.
        {

          // Find the DoFs we want to constrain. There are degree+1 in total.
          // Create a map from these to the face index
          // Note:
          //    - for a single FE_Nedelec<2> element this is
          //      simply 0 to fe.dofs_per_face
          //    - for FESystem<2> this just requires matching the
          //      base element, fe.system_to_base_index.first.first
          //      and the copy of the base element we're interested
          //      in, fe.system_to_base_index.first.second
          std::vector<unsigned int> associated_edge_dof_to_face_dof (degree + 1);

          unsigned int associated_edge_dof_index = 0;
          for (unsigned int face_idx = 0; face_idx < fe.dofs_per_face; ++face_idx)
            {
              const unsigned int cell_idx = fe.face_to_cell_index (face_idx, face);
              if (((dynamic_cast<const FESystem<dim>*> (&fe) != 0)
                   && (fe.system_to_base_index (cell_idx).first == base_indices))
                  || (dynamic_cast<const FE_Nedelec<dim>*> (&fe) != 0)
                  || (dynamic_cast<const MyFE_Nedelec<dim>*> (&fe) != 0))
                {
                  associated_edge_dof_to_face_dof[associated_edge_dof_index] = face_idx;
                  ++associated_edge_dof_index;
                }
            }
          // Sanity check:
          const unsigned int associated_edge_dofs = associated_edge_dof_index;
          Assert (associated_edge_dofs == degree + 1,
                  ExcMessage ("Error: Unexpected number of 2D edge DoFs"));

          // Matrix and RHS vectors to store:
          // We have (degree+1) edge basis functions
          FullMatrix<double> edge_matrix (degree + 1,degree + 1);
          FullMatrix<double> edge_matrix_inv (degree + 1,degree + 1);
          Vector<double> edge_rhs (degree + 1);
          Vector<double> edge_solution (degree + 1);

          const FEValuesExtractors::Vector vec (first_vector_component);

          // coordinate directions of the face.
          const unsigned int
          face_coordinate_direction[GeometryInfo<dim>::faces_per_cell]
            = { 1, 1, 0, 0 };

          const double tol = 0.5 * cell->face (face)->diameter () / cell->get_fe ().degree;
          const std::vector<Point<dim> > &
          reference_quadrature_points = fe_face_values.get_quadrature_points();//.get_points ();

          // Project the boundary function onto the shape functions for this edge
          // and set up a linear system of equations to get the values for the DoFs
          // associated with this edge.
          for (unsigned int q_point = 0;
               q_point < fe_face_values.n_quadrature_points; ++q_point)
            {
/*              
              // Compute the tangent of the face
              // at the quadrature point.
              Point<dim> shifted_reference_point_1 = reference_quadrature_points[q_point];
              Point<dim> shifted_reference_point_2 = reference_quadrature_points[q_point];

              shifted_reference_point_1 (face_coordinate_direction[face])
              += tol;
              shifted_reference_point_2 (face_coordinate_direction[face])
              -= tol;
              Tensor<1,dim> tangential
                = 0.5*( (fe_face_values.get_mapping ()
                   .transform_unit_to_real_cell (cell,
                                                 shifted_reference_point_1)
                   -
                   fe_face_values.get_mapping ()
                   .transform_unit_to_real_cell (cell,
                                                 shifted_reference_point_2))
                  / tol );
                double tangnorm =tangential.norm ();
                Tensor<1,dim> oldtangential = tangential;
              tangential /= tangential.norm ();
              
              std::cout << q_point << " | " << reference_quadrature_points[q_point] << " | "
              << shifted_reference_point_1 << " | " << shifted_reference_point_2 << " | "
              <<  oldtangential << " | " << tangnorm  << " | " << tangential << std::endl;
*/
              // Compute the entires of the linear system
              // Note the system is symmetric so we could only compute the lower/upper triangle.
              //
              // The matrix entries are
              // \int_{edge} (tangential * edge_shape_function_i) * (tangential * edge_shape_function_j) dS
              //
              // The RHS entries are:
              // \int_{edge} (tangential* boundary_value) * (tangential * edge_shape_function_i) dS.
              Point<dim> normal_at_quad_point = fe_face_values.normal_vector(q_point);
              for (unsigned int j = 0; j < associated_edge_dofs; ++j)
                {
                  const unsigned int j_face_idx = associated_edge_dof_to_face_dof[j];
                  const unsigned int j_cell_idx = fe.face_to_cell_index (j_face_idx, face);
                  
                  Tensor<1,dim> phi_j = fe_face_values[vec].value (j_cell_idx, q_point);
                  
                  for (unsigned int i = 0; i < associated_edge_dofs; ++i)
                    {
                      const unsigned int i_face_idx = associated_edge_dof_to_face_dof[i];
                      const unsigned int i_cell_idx = fe.face_to_cell_index (i_face_idx, face);
                      
                      Tensor<1,dim> phi_i = fe_face_values[vec].value (i_cell_idx, q_point);
                      /* leave out tangent
                      edge_matrix(i,j)
                      += fe_face_values.JxW (q_point)
                         * (fe_face_values[vec].value (i_cell_idx, q_point) * tangential)
                         * (fe_face_values[vec].value (j_cell_idx, q_point) * tangential);
                         */
                      edge_matrix(i,j)
                      += fe_face_values.JxW (q_point) 
                         *( (phi_i[1]*normal_at_quad_point(0) - phi_i[0]*normal_at_quad_point(1))
                         * (phi_j[1]*normal_at_quad_point(0) - phi_j[0]*normal_at_quad_point(1)) );
                    }
                    /* leave out tangent
                  edge_rhs(j)
                  += fe_face_values.JxW (q_point)
                     * (values[q_point] (first_vector_component) * tangential [0]
                        + values[q_point] (first_vector_component + 1) * tangential [1])
                     * (fe_face_values[vec].value (j_cell_idx, q_point) * tangential);
                     */
                    edge_rhs(j)
                  += fe_face_values.JxW (q_point)
                  *( (values[q_point] (first_vector_component+1) * normal_at_quad_point (0)
                      - values[q_point] (first_vector_component) * normal_at_quad_point (1) )
                  * (phi_j[1]*normal_at_quad_point(0) - phi_j[0]*normal_at_quad_point(1)) );
                }
            }

          // Invert linear system
          edge_matrix_inv.invert(edge_matrix);
          edge_matrix_inv.vmult(edge_solution,edge_rhs);

          // Store computed DoFs
          for (unsigned int associated_edge_dof_index = 0; associated_edge_dof_index < associated_edge_dofs; ++associated_edge_dof_index)
            {
              dof_values[associated_edge_dof_to_face_dof[associated_edge_dof_index]] = edge_solution (associated_edge_dof_index);
              dofs_processed[associated_edge_dof_to_face_dof[associated_edge_dof_index]] = true;
            }
          break;
        }

        case 3:
        {
          const FEValuesExtractors::Vector vec (first_vector_component);

          // First group DoFs associated with edges which we already know.
          // Sort these into groups of dofs (0 -> degree+1 of them) by each edge.
          // This will help when computing the residual for the face projections.
          //
          // This matches with the search done in compute_edge_projection.
          const unsigned int lines_per_face = GeometryInfo<dim>::lines_per_face;
          std::vector<std::vector<unsigned int> >
          associated_edge_dof_to_face_dof(lines_per_face, std::vector<unsigned int> (degree + 1));
          std::vector<unsigned int> associated_edge_dofs (lines_per_face);

          for (unsigned int line = 0; line < lines_per_face; ++line)
            {
              // Lowest DoF in the base element allowed for this edge:
              const unsigned int lower_bound =
                fe.base_element(base_indices.first).face_to_cell_index(line * (degree + 1), face);
              // Highest DoF in the base element allowed for this edge:
              const unsigned int upper_bound =
                fe.base_element(base_indices.first).face_to_cell_index((line + 1) * (degree + 1) - 1, face);
              unsigned int associated_edge_dof_index = 0;
              for (unsigned int line_idx = 0; line_idx < fe.dofs_per_line; ++line_idx)
                {
                  const unsigned int face_idx = line*fe.dofs_per_line + line_idx;
                  const unsigned int cell_idx = fe.face_to_cell_index(face_idx, face);
                  // Check this cell_idx belongs to the correct base_element, component and line:
                  if (((dynamic_cast<const FESystem<dim>*> (&fe) != 0)
                       && (fe.system_to_base_index (cell_idx).first == base_indices)
                       && (lower_bound <= fe.system_to_base_index (cell_idx).second)
                       && (fe.system_to_base_index (cell_idx).second <= upper_bound))
                      || (
                        ( (dynamic_cast<const MyFE_Nedelec<dim>*> (&fe) != 0)
                        || (dynamic_cast<const FE_Nedelec<dim>*> (&fe) != 0) )
                          && (line * (degree + 1) <= face_idx)
                          && (face_idx <= (line + 1) * (degree + 1) - 1)))
                    {
                      associated_edge_dof_to_face_dof[line][associated_edge_dof_index] = face_idx;
                      ++associated_edge_dof_index;
                    }
                }
              // Sanity check:
              associated_edge_dofs[line] = associated_edge_dof_index;
              Assert (associated_edge_dofs[line] == degree + 1,
                      ExcMessage ("Error: Unexpected number of 3D edge DoFs"));
            }

          // Next find the face DoFs associated with the vector components
          // we're interested in. There are 2*degree*(degree+1) DoFs associated
          // with each face (not including edges!).
          //
          // Create a map mapping from the consecutively numbered associated_dofs
          // to the face DoF (which can be transferred to a local cell index).
          //
          // For FE_Nedelec<3> we just need to have a face numbering greater than
          // the number of edge DoFs (=lines_per_face*(degree+1).
          //
          // For FE_System<3> we need to base the base_indices (base element and
          // copy within that base element) and ensure we're above the number of
          // edge DoFs within that base element.
          std::vector<unsigned int> associated_face_dof_to_face_dof (2*degree*(degree + 1));

          // Skip the edge DoFs, so we start at lines_per_face*(fe.dofs_per_line).
          unsigned int associated_face_dof_index = 0;
          for (unsigned int face_idx = lines_per_face*(fe.dofs_per_line); face_idx < fe.dofs_per_face; ++face_idx)
            {
              const unsigned int cell_idx = fe.face_to_cell_index (face_idx, face);
              if (((dynamic_cast<const FESystem<dim>*> (&fe) != 0)
                   && (fe.system_to_base_index (cell_idx).first == base_indices))
                  || ((dynamic_cast<const FE_Nedelec<dim>*> (&fe) != 0))
                  || ((dynamic_cast<const MyFE_Nedelec<dim>*> (&fe) != 0)))
                {
                  associated_face_dof_to_face_dof[associated_face_dof_index] = face_idx;
                  ++associated_face_dof_index;
                }
            }
          // Sanity check:
          const unsigned int associated_face_dofs = associated_face_dof_index;
          Assert (associated_face_dofs == 2*degree*(degree + 1),
                  ExcMessage ("Error: Unexpected number of 3D face DoFs"));

          // Storage for the linear system.
          // There are 2*degree*(degree+1) DoFs associated with a face in 3D.
          // Note this doesn't include the DoFs associated with edges on that face.
          FullMatrix<double> face_matrix (2*degree*(degree + 1));
          FullMatrix<double> face_matrix_inv (2*degree*(degree + 1));
          Vector<double> face_rhs (2*degree*(degree + 1));
          Vector<double> face_solution (2*degree*(degree + 1));

          // Project the boundary function onto the shape functions for this face
          // and set up a linear system of equations to get the values for the DoFs
          // associated with this face. We also must include the residuals from the
          // shape funcations associated with edges.
          Tensor<1, dim> tmp;
          Tensor<1, dim> normal_vector,
                 cross_product_i,
                 cross_product_j,
                 cross_product_rhs;

          // Store all normal vectors at quad points:
          std::vector<Point<dim> > normal_vector_list(fe_face_values.get_normal_vectors());

          // Loop to construct face linear system.
          for (unsigned int q_point = 0;
               q_point < fe_face_values.n_quadrature_points; ++q_point)
            {
              // First calculate the residual from the edge functions
              // store the result in tmp.
              //
              // Edge_residual =
              //        boundary_value - (
              //            \sum_(edges on face)
              //                 \sum_(DoFs on edge) edge_dof_value*edge_shape_function
              //                   )
              for (unsigned int d = 0; d < dim; ++d)
                {
                  tmp[d]=0.0;
                }
              for (unsigned int line = 0; line < lines_per_face; ++line)
                {
                  for (unsigned int associated_edge_dof = 0; associated_edge_dof < associated_edge_dofs[line]; ++associated_edge_dof)
                    {
                      const unsigned int face_idx = associated_edge_dof_to_face_dof[line][associated_edge_dof];
                      const unsigned int cell_idx = fe.face_to_cell_index(face_idx, face);
                      tmp -= dof_values[face_idx]*fe_face_values[vec].value(cell_idx, q_point);
                    }
                }

              for (unsigned int d=0; d<dim; ++d)
                {
                  tmp[d] += values[q_point] (first_vector_component + d);
                }

              // Tensor of normal vector on the face at q_point;
              for (unsigned int d = 0; d < dim; ++d)
                {
                  normal_vector[d] = normal_vector_list[q_point](d);
                }
              // Now compute the linear system:
              // On a face:
              // The matrix entries are:
              // \int_{face} (n x face_shape_function_i) \cdot ( n x face_shape_function_j) dS
              //
              // The RHS entries are:
              // \int_{face} (n x (Edge_residual) \cdot (n x face_shape_function_i) dS

              for (unsigned int j = 0; j < associated_face_dofs; ++j)
                {
                  const unsigned int j_face_idx = associated_face_dof_to_face_dof[j];
                  const unsigned int cell_j = fe.face_to_cell_index (j_face_idx, face);

                  cross_product(cross_product_j,
                                normal_vector,
                                fe_face_values[vec].value(cell_j, q_point));

                  for (unsigned int i = 0; i < associated_face_dofs; ++i)
                    {
                      const unsigned int i_face_idx = associated_face_dof_to_face_dof[i];
                      const unsigned int cell_i = fe.face_to_cell_index (i_face_idx, face);
                      cross_product(cross_product_i,
                                    normal_vector,
                                    fe_face_values[vec].value(cell_i, q_point));

                      face_matrix(i,j)
                      += fe_face_values.JxW (q_point)
                         *cross_product_i
                         *cross_product_j;

                    }
                  // compute rhs
                  cross_product(cross_product_rhs,
                                normal_vector,
                                tmp);
                  face_rhs(j)
                  += fe_face_values.JxW (q_point)
                     *cross_product_rhs
                     *cross_product_j;

                }
            }

          // Solve lienar system:
          face_matrix_inv.invert(face_matrix);
          face_matrix_inv.vmult(face_solution, face_rhs);


          // Store computed DoFs:
          for (unsigned int associated_face_dof = 0; associated_face_dof < associated_face_dofs; ++associated_face_dof)
            {
              dof_values[associated_face_dof_to_face_dof[associated_face_dof]] = face_solution(associated_face_dof);
              dofs_processed[associated_face_dof_to_face_dof[associated_face_dof]] = true;
            }
          break;
        }
        default:
          Assert (false, ExcNotImplemented ());
        }
    }

    template <int dim, class DH>
    void
    compute_project_boundary_values_curl_conforming_l2 (const DH &dof_handler,
                                                        const unsigned int first_vector_component,
                                                        const Function<dim> &boundary_function,
                                                        const types::boundary_id boundary_component,
                                                        ConstraintMatrix &constraints,
                                                        const hp::MappingCollection<dim, dim> &mapping_collection)
    {
      // L2-projection based interpolation formed in one (in 2D) or two (in 3D) steps.
      //
      // In 2D we only need to constrain edge DoFs.
      //
      // In 3D we need to constrain both edge and face DoFs. This is done in two parts.
      //
      // For edges, since the face shape functions are zero here ("bubble functions"),
      // we project the tangential component of the boundary function and compute
      // the L2-projection. This returns the values for the DoFs associated with each edge shape
      // function. In 3D, this is computed by internals::compute_edge_projection_l2, in 2D,
      // it is handled by compute_face_projection_curl_conforming_l2.
      //
      // For faces we compute the residual of the boundary function which is satisfied
      // by the edge shape functions alone. Which can then be used to calculate the
      // remaining face DoF values via a projection which leads to a linear system to
      // solve. This is handled by compute_face_projection_curl_conforming_l2
      //
      // For details see (for example) section 4.2:
      // Electromagnetic scattering simulation using an H (curl) conforming hp finite element
      // method in three dimensions, PD Ledger, K Morgan, O Hassan,
      // Int. J.  Num. Meth. Fluids, Volume 53, Issue 8, pages 1267–1296, 20 March 2007:
      // http://onlinelibrary.wiley.com/doi/10.1002/fld.1223/abstract

      // Create hp FEcollection, dof_handler can be either hp or standard type.
      // From here on we can treat it like a hp-namespace object.
      const hp::FECollection<dim> fe_collection (dof_handler.get_fe ());

      // Create face quadrature collection
      hp::QCollection<dim - 1> face_quadrature_collection;
      for (unsigned int i = 0; i < fe_collection.size (); ++i)
        {
          const QGauss<dim - 1>  reference_face_quadrature (2 * fe_collection[i].degree + 3);
          face_quadrature_collection.push_back(reference_face_quadrature);
        }

      hp::FEFaceValues<dim> fe_face_values (mapping_collection, fe_collection,
                                            face_quadrature_collection,
                                            update_values |
                                            update_quadrature_points |
                                            update_normal_vectors |
                                            update_JxW_values);

      // Storage for dof values found and whether they have been processed:
      std::vector<bool> dofs_processed;
      std::vector<double> dof_values;
      std::vector<types::global_dof_index> face_dof_indices;
      typename DH::active_cell_iterator cell = dof_handler.begin_active ();

      switch (dim)
        {
        case 2:
        {
          for (; cell != dof_handler.end (); ++cell)
            {
              if (cell->at_boundary () && cell->is_locally_owned ())
                {
                  for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
                    {
                      if (cell->face (face)->boundary_id () == boundary_component)
                        {
                          // If the FE is an FE_Nothing object there is no work to do
                          if (dynamic_cast<const FE_Nothing<dim>*> (&cell->get_fe ()) != 0)
                            {
                              return;
                            }

                          // This is only implemented for FE_Nedelec elements.
                          // If the FE is a FESystem we cannot check this.
                          if (dynamic_cast<const FESystem<dim>*> (&cell->get_fe ()) == 0)
                            {
                              typedef FiniteElement<dim> FEL;
                              AssertThrow (
                                ((dynamic_cast<const FE_Nedelec<dim>*> (&cell->get_fe ()) != 0)
                                || (dynamic_cast<const MyFE_Nedelec<dim>*> (&cell->get_fe ()) != 0)),
                                           typename FEL::ExcInterpolationNotImplemented ());
                            }

                          const unsigned int dofs_per_face = cell->get_fe ().dofs_per_face;

                          dofs_processed.resize (dofs_per_face);
                          dof_values.resize (dofs_per_face);

                          for (unsigned int dof = 0; dof < dofs_per_face; ++dof)
                            {
                              dof_values[dof] = 0.0;
                              dofs_processed[dof] = false;
                            }

                          // Compute the projection of the boundary function on the edge.
                          // In 2D this is all that's required.
                          compute_face_projection_curl_conforming_l2 (cell, face, fe_face_values,
                                                                      boundary_function,
                                                                      first_vector_component,
                                                                      dof_values, dofs_processed);

                          // store the local->global map:
                          face_dof_indices.resize(dofs_per_face);
                          cell->face (face)->get_dof_indices (face_dof_indices,
                                                              cell->active_fe_index ());

                          // Add the computed constraints to the constraint matrix,
                          // assuming the degree of freedom is not already constrained.
                          for (unsigned int dof = 0; dof < dofs_per_face; ++dof)
                            {
                              if (dofs_processed[dof] && constraints.can_store_line (face_dof_indices[dof])
                                  && !(constraints.is_constrained (face_dof_indices[dof])))
                                {
                                  constraints.add_line (face_dof_indices[dof]);
                                  if (std::abs (dof_values[dof]) > 1e-15)
                                    {
                                      constraints.set_inhomogeneity (face_dof_indices[dof], dof_values[dof]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
          break;
        }

        case 3:
        {
          hp::QCollection<dim> edge_quadrature_collection;

          // Create equivalent of FEEdgeValues:
          for (unsigned int i = 0; i < fe_collection.size (); ++i)
            {
              const QGauss<dim-2> reference_edge_quadrature (2 * fe_collection[i].degree + 3);
              for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
                {
                  for (unsigned int line = 0; line < GeometryInfo<dim>::lines_per_face; ++line)
                    {
                      edge_quadrature_collection.push_back
                      (QProjector<dim>::project_to_face
                       (QProjector<dim - 1>::project_to_face
                        (reference_edge_quadrature, line), face));
                    }
                }
            }

          hp::FEValues<dim> fe_edge_values (mapping_collection, fe_collection,
                                            edge_quadrature_collection,
                                            update_jacobians |
                                            update_JxW_values |
                                            update_quadrature_points |
                                            update_values);

          for (; cell != dof_handler.end (); ++cell)
            {
              if (cell->at_boundary () && cell->is_locally_owned ())
                {
                  for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
                    {
                      if (cell->face (face)->boundary_id () == boundary_component)
                        {
                          // If the FE is an FE_Nothing object there is no work to do
                          if (dynamic_cast<const FE_Nothing<dim>*> (&cell->get_fe ()) != 0)
                            {
                              return;
                            }

                          // This is only implemented for FE_Nedelec elements.
                          // If the FE is a FESystem we cannot check this.
                          if (dynamic_cast<const FESystem<dim>*> (&cell->get_fe ()) == 0)
                            {
                              typedef FiniteElement<dim> FEL;

                              AssertThrow (
                                ( (dynamic_cast<const FE_Nedelec<dim>*> (&cell->get_fe ()) != 0)
                                || (dynamic_cast<const MyFE_Nedelec<dim>*> (&cell->get_fe ()) != 0)),
                                           typename FEL::ExcInterpolationNotImplemented ());
                            }

                          const unsigned int superdegree = cell->get_fe ().degree;
                          const unsigned int degree = superdegree - 1;
                          const unsigned int dofs_per_face = cell->get_fe ().dofs_per_face;

                          dofs_processed.resize (dofs_per_face);
                          dof_values.resize (dofs_per_face);
                          for (unsigned int dof = 0; dof < dofs_per_face; ++dof)
                            {
                              dof_values[dof] = 0.0;
                              dofs_processed[dof] = false;
                            }

                          // First compute the projection on the edges.
                          for (unsigned int line = 0; line < GeometryInfo<3>::lines_per_face; ++line)
                            {
                              compute_edge_projection_l2 (cell, face, line, fe_edge_values,
                                                          boundary_function,
                                                          first_vector_component,
                                                          dof_values, dofs_processed);
                            }

                          // If there are higher order shape functions, then we
                          // still need to compute the face projection
                          if (degree > 0)
                            {
                              compute_face_projection_curl_conforming_l2 (cell, face, fe_face_values,
                                                                          boundary_function,
                                                                          first_vector_component,
                                                                          dof_values,
                                                                          dofs_processed);
                            }

                          // Store the computed values in the global vector.
                          face_dof_indices.resize(dofs_per_face);
                          cell->face (face)->get_dof_indices (face_dof_indices,
                                                              cell->active_fe_index ());

                          for (unsigned int dof = 0; dof < dofs_per_face; ++dof)
                            {
                              if (dofs_processed[dof] && constraints.can_store_line (face_dof_indices[dof])
                                  && !(constraints.is_constrained (face_dof_indices[dof])))
                                {
                                  constraints.add_line (face_dof_indices[dof]);

                                  if (std::abs (dof_values[dof]) > 1e-15)
                                    {
                                      constraints.set_inhomogeneity (face_dof_indices[dof], dof_values[dof]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
          break;
        }
        default:
          Assert (false, ExcNotImplemented ());
        }
    }

  }


  template <int dim>
  void
  project_boundary_values_curl_conforming_l2 (const DoFHandler<dim> &dof_handler,
                                              const unsigned int first_vector_component,
                                              const Function<dim> &boundary_function,
                                              const types::boundary_id boundary_component,
                                              ConstraintMatrix &constraints,
                                              const Mapping<dim> &mapping = StaticMappingQ1<dim>::mapping)
  {
    // non-hp version - calls the internal
    // compute_project_boundary_values_curl_conforming_l2() function
    // above after recasting the mapping.

    hp::MappingCollection<dim> mapping_collection (mapping);
    internals::
    compute_project_boundary_values_curl_conforming_l2(dof_handler,
                                                       first_vector_component,
                                                       boundary_function,
                                                       boundary_component,
                                                       constraints,
                                                       mapping_collection);
  }

  template <int dim>
  void
  project_boundary_values_curl_conforming_l2 (const hp::DoFHandler<dim> &dof_handler,
                                              const unsigned int first_vector_component,
                                              const Function<dim> &boundary_function,
                                              const types::boundary_id boundary_component,
                                              ConstraintMatrix &constraints,
                                              const hp::MappingCollection<dim, dim> &mapping_collection = hp::StaticMappingQ1<dim>::mapping_collection)
  {
    // hp version - calls the internal
    // compute_project_boundary_values_curl_conforming_l2() function above.
    internals::
    compute_project_boundary_values_curl_conforming_l2(dof_handler,
                                                       first_vector_component,
                                                       boundary_function,
                                                       boundary_component,
                                                       constraints,
                                                       mapping_collection);
  }
}
#endif