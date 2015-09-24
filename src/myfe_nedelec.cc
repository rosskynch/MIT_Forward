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

#include <myfe_nedelec.h>

// Constructor:
template <int dim>
MyFE_Nedelec<dim>::MyFE_Nedelec (const unsigned int degree)
//                                  const FiniteElementData<dim> &fe_data,
//                                  const std::vector<bool> &restriction_is_additive_flags,
//                                  const std::vector<ComponentMask> &nonzero_components);
: 
FiniteElement<dim,dim>
(FiniteElementData<dim> (get_dpo_vector (degree), dim, degree + 1,
                         FiniteElementData<dim>::Hcurl, 1),
 std::vector<bool> (compute_n_pols (degree), true),
 std::vector<ComponentMask>(compute_n_pols (degree),
                            std::vector<bool> (dim, true) ) ),
polynomial_space (create_polynomials (degree))
{
  Assert (dim >= 2, ExcImpossibleInDim(dim));

  this->mapping_type = mapping_nedelec;
  // Set up the table converting
  // components to base
  // components. Since we have only
  // one base element, everything
  // remains zero except the
  // component in the base, which is
  // the component itself
  for (unsigned int comp=0; comp<this->n_components() ; ++comp)
  {
    this->component_to_base_table[comp].first.second = comp;
  }
}

// Shape functions:
template <int dim>
double
MyFE_Nedelec<dim>::shape_value (const unsigned int, const Point<dim> &) const

{
  typedef    FiniteElement<dim,dim> FEE;
  Assert(false, typename FEE::ExcFENotPrimitive());
  return 0.;
}



template <int dim>
double
MyFE_Nedelec<dim>::shape_value_component (const unsigned int i,
                                          const Point<dim> &p,
                                          const unsigned int component) const
{
  // Not implemented yet:
  Assert (false, ExcNotImplemented ());
  return 0.;
  /*
  Assert (i<this->dofs_per_cell, ExcIndexRange(i,0,this->dofs_per_cell));
  Assert (component < dim, ExcIndexRange (component, 0, dim));

  if (cached_point != p || cached_values.size() == 0)
    {
      cached_point = p;
      cached_values.resize(poly_space.n());
      poly_space.compute(p, cached_values, cached_grads, cached_grad_grads);
    }

  double s = 0;
  if (inverse_node_matrix.n_cols() == 0)
    return cached_values[i][component];
  else
    for (unsigned int j=0; j<inverse_node_matrix.n_cols(); ++j)
      s += inverse_node_matrix(j,i) * cached_values[j][component];
  return s;
  */
}



template <int dim>
Tensor<1,dim>
MyFE_Nedelec<dim>::shape_grad (const unsigned int,
                               const Point<dim> &) const
{
  typedef    FiniteElement<dim,dim> FEE;
  Assert(false, typename FEE::ExcFENotPrimitive());
  return Tensor<1,dim>();
}



template <int dim>
Tensor<1,dim>
MyFE_Nedelec<dim>::shape_grad_component (const unsigned int i,
                                         const Point<dim> &p,
                                         const unsigned int component) const
{
  // Not implemented yet:
  Assert (false, ExcNotImplemented ());
  return Tensor<1,dim> ();
  /*
  Assert (i<this->dofs_per_cell, ExcIndexRange(i,0,this->dofs_per_cell));
  Assert (component < dim, ExcIndexRange (component, 0, dim));

  if (cached_point != p || cached_grads.size() == 0)
    {
      cached_point = p;
      cached_grads.resize(poly_space.n());
      poly_space.compute(p, cached_values, cached_grads, cached_grad_grads);
    }

  Tensor<1,dim> s;
  if (inverse_node_matrix.n_cols() == 0)
    return cached_grads[i][component];
  else
    for (unsigned int j=0; j<inverse_node_matrix.n_cols(); ++j)
      s += inverse_node_matrix(j,i) * cached_grads[j][component];

  return s;
  */
}



template <int dim>
Tensor<2,dim>
MyFE_Nedelec<dim>::shape_grad_grad (const unsigned int, const Point<dim> &) const
{
  typedef    FiniteElement<dim,dim> FEE;
  Assert(false, typename FEE::ExcFENotPrimitive());
  return Tensor<2,dim>();
}



template <int dim>
Tensor<2,dim>
MyFE_Nedelec<dim>::shape_grad_grad_component (const unsigned int i,
                                              const Point<dim> &p,
                                              const unsigned int component) const
{
  // Not implemented yet:
  Assert (false, ExcNotImplemented ());
  return Tensor<2,dim>();
  /*
  Assert (i<this->dofs_per_cell, ExcIndexRange(i,0,this->dofs_per_cell));
  Assert (component < dim, ExcIndexRange (component, 0, dim));

  if (cached_point != p || cached_grad_grads.size() == 0)
    {
      cached_point = p;
      cached_grad_grads.resize(poly_space.n());
      poly_space.compute(p, cached_values, cached_grads, cached_grad_grads);
    }

  Tensor<2,dim> s;
  if (inverse_node_matrix.n_cols() == 0)
    return cached_grad_grads[i][component];
  else
    for (unsigned int j=0; j<inverse_node_matrix.n_cols(); ++j)
      s += inverse_node_matrix(i,j) * cached_grad_grads[j][component];

  return s;
  */
}


  
template <int dim>
typename Mapping<dim,dim>::InternalDataBase *
MyFE_Nedelec<dim>::get_data (
  const UpdateFlags      update_flags,
  const Mapping<dim,dim>    &mapping,
  const Quadrature<dim> &quadrature) const
{
  InternalData *data = new InternalData;
  data->update_once = update_once(update_flags);
  data->update_each = update_each(update_flags);
  data->update_flags = data->update_once | data->update_each;
  
  // Useful quantities:
  const unsigned int degree (this->degree-1); // not FE holds input degree+1
  
  const unsigned int vertices_per_cell = GeometryInfo<dim>::vertices_per_cell;
  const unsigned int lines_per_face = GeometryInfo<dim>::lines_per_face;
  const unsigned int lines_per_cell = GeometryInfo<dim>::lines_per_cell;
  const unsigned int faces_per_cell = GeometryInfo<dim>::faces_per_cell;
  
  const unsigned int dofs_per_cell = this->dofs_per_cell;
  const unsigned int dofs_per_line = this->dofs_per_line;  
  const unsigned int dofs_per_quad = this->dofs_per_quad;
  const unsigned int dofs_per_face = this->dofs_per_face;
  const unsigned int dofs_per_hex = this->dofs_per_hex;
  
  const unsigned int n_line_dofs = dofs_per_line*lines_per_cell;
  const unsigned int n_face_dofs = dofs_per_quad*faces_per_cell;
  const unsigned int n_cell_dofs = (dim == 2) ? dofs_per_quad : dofs_per_hex; //TODO create switch for dim=2/3
    
  const UpdateFlags flags(data->update_flags);
  const unsigned int n_q_points = quadrature.size();
  
  // In all cases we want to fill the values of sigma & lambda in the
  // internal data class.
  // TODO: decide if this is needed.
  // Resize the storage:
  data->sigma.resize(n_q_points);
  data->sigma_imj_values.resize(n_q_points);  
  data->lambda.resize(n_q_points);
  data->lambda_ipj.resize(n_q_points);
  for (unsigned int q=0; q<n_q_points; ++q)
  {
    data->sigma[q].resize(vertices_per_cell);
    data->sigma_imj_values[q].resize(vertices_per_cell);
    data->lambda[q].resize(vertices_per_cell);
    data->lambda_ipj[q].resize(vertices_per_cell);  
    for (unsigned int i=0; i<vertices_per_cell; ++i)
    {
      data->sigma_imj_values[q][i].resize(vertices_per_cell);
      data->lambda_ipj[q][i].resize(vertices_per_cell);      
    }
  }
  data->sigma_imj_grads.resize(vertices_per_cell);
  data->sigma_imj_sign.resize(vertices_per_cell);
  data->sigma_imj_component.resize(vertices_per_cell);
  for (unsigned int i=0; i<vertices_per_cell; ++i)
  {
    data->sigma_imj_grads[i].resize(vertices_per_cell);
    data->sigma_imj_sign[i].resize(vertices_per_cell);
    data->sigma_imj_component[i].resize(vertices_per_cell);
    for (unsigned int j=0; j<vertices_per_cell; ++j)
    {
      data->sigma_imj_grads[i][j].resize(dim);
    }
  }
  
  // Resize shape function arrays according to update flags:
  if (flags & update_values)
  {
    data->shape_values.resize (dofs_per_cell);
    for (unsigned int i=0; i<dofs_per_cell; ++i)
    {
      data->shape_values[i].resize (n_q_points);
    }
  }
  
  if (flags & update_gradients)
  {
    data->shape_grads.resize(dofs_per_cell);
    for (unsigned int i=0; i<dofs_per_cell; ++i)
    {
      data->shape_grads[i].resize (n_q_points);
    }
  }
  // Not implementing second derivatives yet:
  if (flags & update_hessians)
  {
    Assert(false, ExcNotImplemented());
  }
  
  std::vector< Point<dim> > p_list(n_q_points);
  p_list = quadrature.get_points();
  
  
  switch (dim)
  {
    case 2:
    {
      // Compute values of sigma & lambda and the sigma differences and lambda additions.
      for (unsigned int q=0; q<n_q_points; ++q)
      {
        data->sigma[q][0] = (1.0 - p_list[q][0]) + (1.0 - p_list[q][1]);
        data->sigma[q][1] = p_list[q][0] + (1.0 - p_list[q][1]);
        data->sigma[q][2] = (1.0 - p_list[q][0]) + p_list[q][1];
        data->sigma[q][3] = p_list[q][0] + p_list[q][1];
        
        data->lambda[q][0] = (1.0 - p_list[q][0])*(1.0 - p_list[q][1]);
        data->lambda[q][1] = p_list[q][0]*(1.0 - p_list[q][1]);
        data->lambda[q][2] = (1.0 - p_list[q][0])*p_list[q][1];
        data->lambda[q][3] = p_list[q][0]*p_list[q][1];
        for (unsigned int i=0; i<vertices_per_cell; ++i)
        {
          for (unsigned int j=0; j<vertices_per_cell; ++j)
          {
            data->sigma_imj_values[q][i][j] = data->sigma[q][i] - data->sigma[q][j]; 
            data->lambda_ipj[q][i][j] = data->lambda[q][i] + data->lambda[q][j];
          }
        }
      }
      
      
      // Now fill edgeDoF_to_poly
      // TODO CHECK THIS IS RIGHT (first check OK.. will do one more).
      //
      // polyspace is Lx_i (x) Ly_j, 0 <= i,j < degree+2
      //
      // To access Lx_i (x) Ly_j from poly_space
      // then we want to access the result of poly_space.compute
      // The corresponding vector entry is: [i + j*(degree+2)].
      data->edgeDoF_to_poly.resize(lines_per_cell);
      for (unsigned int m=0; m<lines_per_cell; ++m)
      {
        data->edgeDoF_to_poly[m].resize(dofs_per_line);
      }
      for (unsigned int m=0; m<2; ++m)
      {
        // Lowest order edges 0/1 are Lx0 (x) Ly1. (NOT CURRENTLY USED).
        data->edgeDoF_to_poly[m][0]=(degree+2);
        // Higher order edges 0 and 1 have fixed Px:
        // want grad( Lx1 (x) Ly(j+1)), j=1,..,degree.
        const unsigned int i=1;
        for (unsigned int j=0; j<degree; ++j)
        {      
          data->edgeDoF_to_poly[m][j+1] = i + (j+2)*(degree+2); 
        }
      }
      for (unsigned int m=2; m<4; ++m)
      {    
        // Lowest order edges 2/3 are Lx1 (x) Ly0.
        data->edgeDoF_to_poly[m][0]=1;
        // Higher order edges 0 and 1 have fixed Py:
        // want grad( Lx(i+1) (x) Ly1), i=1,..,degree.
        const unsigned int j=1;
        for (unsigned int i=0; i<degree; ++i)
        {
          data->edgeDoF_to_poly[m][i+1] = (i+2) + j*(degree+2);
        }
      }
      
      // For now we only precompute the cell-based shape functions
      // TODO: However, we could also precompute the edge-based ones
      // on the "standard" cell and store them. Then by simply comparing
      // the local edge numberings, we could avoid recomputing in fill_fe*_values.
      //  
      // Note: the local dof numberings follow the usual order of lines -> faces -> cells
      //       (we have no vertex-based DoFs in this element).
      // For a given cell we have:
      //      n_line_dofs = dofs_per_line*lines_per_cell.
      //      n_face_dofs = dofs_per_face*faces_per_cell.
      //      n_cell_dofs = dofs_per_quad (2D)
      //                  = dofs_per_hex (3D)
      //
      // i.e. For the local dof numbering:
      //      the first line dof is 0,
      //      the first face dof is n_line_dofs,
      //      the first cell dof is n_line_dofs + n_face_dofs.
      //
      // On a line, DoFs are ordered first by line_dof and then line_index:
      // i.e. line_dof_index = line_dof + line_index*(dofs_per_line)
      //
      // and similarly for faces:
      // i.e. face_dof_index = face_dof + face_index*(dofs_per_face).
      //
      // HOWEVER, we have different types of DoFs on a line/face/cell.
      // On a line we have two types, lowest order and higher order gradients.
      //    - The numbering is such the lowest order is first, then higher order.
      //      This is simple enough as there is only 1 lowest order and degree higher
      //      orders DoFs per line.
      //
      // TODO: Add faces (in 3D).
      //
      // On a cell (in 2D, TODO: include 3D), we have 3 types: Type 1/2/3:
      //    - The ordering done by type:
      //      - Type 1: 0 <= i1,j1 < degree. degree^2 in total.
      //        Numbered: ij1 = i1 + j1*(degree).        i.e. cell_dof_index = ij1.
      //      - Type 2: 0 <= i2,j2 < degree. degree^2 in total.
      //        Numbered: ij2 = i2 + j2*(degree).        i.e. cell_dof_index = degree^2 + ij2
      //      - Type 3: 0 <= i3 < 2*degree. 2*degree in total.
      //        Numbered: ij3 = i3.                      i.e. cell_dof_index =  2*(degree^2) + ij3.
      //
      // These then fit into the local dof numbering described above:
      // - local dof numberings are:
      //   line_dofs: local_dof = line_dof_index.    0 <= local_dof < dofs_per_line*lines_per_cell
      //   face_dofs: local_dof = n_line_dofs*lines_per_cell + face_dof_index.
      //   cell dofs: local_dof = n_lines_dof + n_face_dofs + cell_dof_index.
      
      if (flags & update_values | update_gradients)
      {
        // scratch arrays:
        std::vector<double> values(0); // Not needed, so leave empty.
        std::vector<Tensor<1,dim>> grads(0);
        std::vector<Tensor<2,dim>> grad_grads(0);
        if (flags & update_values)
        {
          grads.resize(polynomial_space.n ());
        }
        if (flags & update_gradients)
        {
          grad_grads.resize(polynomial_space.n ());
        }
        if (degree>0) // If degree=0 then nothing to precompute
        {   
          std::vector< Point<dim> > cell_points (n_q_points);
          for (unsigned int q=0; q<n_q_points; ++q)
          {
            for (unsigned int d=0; d<dim; ++d)
            {
              cell_points[q][d] = 2.0*p_list[q][d] - 1.0;
            }
          }
          // Cell-based shape functions:
          //
          // Type 1 (gradients):
          // \phi^{C_{1}}_{ij) = grad( L_{i+2}(2x-1)L_{j+2}(2y-1) ),
          //
          // 0 <= i,j < degree.
          //
          // which gives 2* the grad produced at a point (2x-1, 2y-1) 
          // by polynomial_space[i+2 +(j+2)*(degree+2)], 
          
          //TODO: include n_face_dofs for 3D case.
          //      Can set n_face_dofs=0 for 2D ??
          unsigned int cell_type1_offset = n_line_dofs; 
          
          // Type 2:
          // \phi^{C_{2}}_{ij) = L'_{i+2}(2x-1) L_{j+2}(2y-1) \mathbf{e}_{x}
          //                     L_{i+2}(2x-1) L'_{j+2}(2y-1) \mathbf{e}_{y},
          //
          // 0 <= i,j < degree.
          
          unsigned int cell_type2_offset = n_line_dofs + degree*degree;
          
          // Type 3 (two subtypes):
          // \phi^{C_{3}}_{i)        = L_{i+2}(2y-1) \mathbf{e}_{x}
          //
          // \phi^{C_{3}}_{i+degree) = L_{i+2}(2x-1) \mathbf{e}_{y}
          //
          // 0 <= i < degree
          unsigned int cell_type3_offset = n_line_dofs + 2*degree*degree;
          
          // Loop through quad points:
          //
          // TODO: Large amounts of this could be avoided if we
          //       had a MyPolynomialsNedelec which generated the shape functions very carefully,
          //       in the correct order. I.e. the polynomial_space produces vector-valued values, etc.
          //       See the current PolynomialsNedelec as an example.
          
          for (unsigned int q=0; q<n_q_points; ++q)
          {
            polynomial_space.compute(cell_points[q], values, grads, grad_grads);
            if (flags & update_values)
            {
              // TODO merge all 3 types into a single set of nested loops over dof_i/dof_j.
              for (unsigned int dof_j=0; dof_j<degree; ++dof_j)
              {
                const unsigned int dof_shift1 (cell_type1_offset + dof_j*degree);
                const unsigned int poly_shift1 ((dof_j+2)*(degree+2));
                for (unsigned int dof_i=0; dof_i<degree; ++dof_i)
                {
                  const unsigned int dof_index1 (dof_i + dof_shift1);
                  const unsigned int poly_index (dof_i+2 + poly_shift1);
                  
                  data->shape_values[dof_index1][q] = 2.0*grads[poly_index]; // *2 to account for Point (2x-1, 2y-1).
                }
              }
              for (unsigned int dof_j=0; dof_j<degree; ++dof_j)
              {
                const unsigned int dof_shift2 (cell_type2_offset + dof_j*degree);
                
                // Store the value and 1st derivative of L_{j+2} at cell_points[q].
                // polyi[d][n] stores the nth derivative of L_{j+2} at cell_points[q][d]
                // i.e. 2x-1, d = 0. 2y-1 for d=1.
                std::vector<std::vector<double> > polyj(dim, std::vector<double> (2));
                for (unsigned int d=0; d<dim; ++d)
                {
                  polynomials_1d[dof_j+2].value(cell_points[q][d], polyj[d]);
                }
                // Type 2:
                for (unsigned int dof_i=0; dof_i<degree; ++dof_i)
                {
                  // Store the value and 1st derivative of L_{i+2} at cell_points[q].
                  // polyi[d][n] stores the nth derivative of L_{i+2} at cell_points[q][d]
                  // i.e. 2x-1, d = 0. 2y-1 for d=1
                  std::vector<std::vector<double> > polyi(dim, std::vector<double> (2));
                  for (unsigned int d=0; d<dim; ++d)
                  {
                    polynomials_1d[dof_i+2].value(cell_points[q][d], polyi[d]);
                  }
                  const unsigned int dof_index2 (dof_shift2 + dof_i);
                  
                  data->shape_values[dof_index2][q][0] =   polyi[0][1]*polyj[1][0];
                  data->shape_values[dof_index2][q][1] = - polyi[0][0]*polyj[1][1];
                }
                // Type 3:
                const unsigned int dof_index3_x(cell_type3_offset + dof_j);
                const unsigned int dof_index3_y(cell_type3_offset + dof_j + degree);
                
                data->shape_values[dof_index3_x][q][0] = polyj[1][0];
                data->shape_values[dof_index3_x][q][1] = 0.0;
                
                data->shape_values[dof_index3_y][q][0] = 0.0;
                data->shape_values[dof_index3_y][q][1] = polyj[0][0];
              }
            }
            if (flags & update_gradients)
            {
              for (unsigned int dof_j=0; dof_j<degree; ++dof_j)
              {
                const unsigned int dof_shift1 (cell_type1_offset + dof_j*degree);
                const unsigned int poly_shift1 ((dof_j+2)*(degree+2));
                for (unsigned int dof_i=0; dof_i<degree; ++dof_i)
                {
                  const unsigned int dof_index1 (dof_shift1 + dof_i);
                  const unsigned int poly_index (poly_shift1 + dof_i+2);
                  data->shape_grads[dof_index1][q] = 2.0*2.0*grad_grads[poly_index]; // *2^2 to account for Point (2x-1, 2y-1).
                }
              }  
              for (unsigned int dof_j=0; dof_j<degree; ++dof_j)
              {
                const unsigned int dof_shift2 (cell_type2_offset + dof_j*degree);
                
                // Store the value and 1st derivative of L_{j+2} at cell_points[q].
                // polyi[d][n] stores the nth derivative of L_{j+2} at cell_points[q][d]
                // i.e. 2x-1, d = 0. 2y-1 for d=1.
                std::vector<std::vector<double> > polyj(dim, std::vector<double> (3));
                // TODO: Could avoid computing the above multiple times.
                //       Maybe introduce n_derivatives=2 or 3 and re-use for shape values and grads.
                for (unsigned int d=0; d<dim; ++d)
                {
                  polynomials_1d[dof_j+2].value(cell_points[q][d], polyj[d]);
                }
                // Type 2:
                for (unsigned int dof_i=0; dof_i<degree; ++dof_i)
                {
                  // Store the value and 1st derivative of L_{i+2} at cell_points[q].
                  // polyi[d][n] stores the nth derivative of L_{i+2} at cell_points[q][d]
                  // i.e. 2x-1, d = 0. 2y-1 for d=1
                  std::vector<std::vector<double> > polyi(dim, std::vector<double> (3));
                  for (unsigned int d=0; d<dim; ++d)
                  {
                    polynomials_1d[dof_i+2].value(cell_points[q][d], polyi[d]);
                  }
                  
                  const unsigned int dof_index2 (dof_shift2 + dof_i);
                  
                  data->shape_grads[dof_index2][q][0][0] =  2.0*polyi[0][2]*polyj[1][0];
                  data->shape_grads[dof_index2][q][0][1] =  2.0*polyi[0][1]*polyj[1][1];
                  data->shape_grads[dof_index2][q][1][0] = -2.0*polyi[0][1]*polyj[1][1];
                  data->shape_grads[dof_index2][q][1][1] = -2.0*polyi[0][0]*polyj[1][2];
                }
                // Type 3:
                const unsigned int dof_index3_x(cell_type3_offset + dof_j);
                const unsigned int dof_index3_y(cell_type3_offset + dof_j + degree);
                
                data->shape_grads[dof_index3_x][q][0][0] = 0.0;
                data->shape_grads[dof_index3_x][q][0][1] = 2.0*polyj[1][1];
                data->shape_grads[dof_index3_x][q][1][0] = 0.0;
                data->shape_grads[dof_index3_x][q][1][1] = 0.0;
                
                data->shape_grads[dof_index3_y][q][0][0] = 0.0;
                data->shape_grads[dof_index3_y][q][0][1] = 0.0;
                data->shape_grads[dof_index3_y][q][1][0] = 2.0*polyj[0][1];
                data->shape_grads[dof_index3_y][q][1][1] = 0.0;
              }
            }
          }
        }
      }
      break;
    }
    case 3:
    {
      // TODO: decide if these need to be stored or just computed once here.
      // Compute values of sigma & lambda and the sigma differences and lambda additions.
      for (unsigned int q=0; q<n_q_points; ++q)
      {
        data->sigma[q][0] = (1.0 - p_list[q][0]) + (1.0 - p_list[q][1]) + (1 - p_list[q][2]);
        data->sigma[q][1] = p_list[q][0] + (1.0 - p_list[q][1]) + (1 - p_list[q][2]);
        data->sigma[q][2] = (1.0 - p_list[q][0]) + p_list[q][1] + (1 - p_list[q][2]);
        data->sigma[q][3] = p_list[q][0] + p_list[q][1] + (1 - p_list[q][2]);
        data->sigma[q][4] = (1.0 - p_list[q][0]) + (1.0 - p_list[q][1]) + p_list[q][2];
        data->sigma[q][5] = p_list[q][0] + (1.0 - p_list[q][1]) + p_list[q][2];
        data->sigma[q][6] = (1.0 - p_list[q][0]) + p_list[q][1] + p_list[q][2];
        data->sigma[q][7] = p_list[q][0] + p_list[q][1] + p_list[q][2];        
        
        data->lambda[q][0] = (1.0 - p_list[q][0])*(1.0 - p_list[q][1])*(1.0 - p_list[q][2]);
        data->lambda[q][1] = p_list[q][0]*(1.0 - p_list[q][1])*(1.0 - p_list[q][2]);
        data->lambda[q][2] = (1.0 - p_list[q][0])*p_list[q][1]*(1.0 - p_list[q][2]);
        data->lambda[q][3] = p_list[q][0]*p_list[q][1]*(1.0 - p_list[q][2]);
        data->lambda[q][4] = (1.0 - p_list[q][0])*(1.0 - p_list[q][1])*p_list[q][2];
        data->lambda[q][5] = p_list[q][0]*(1.0 - p_list[q][1])*p_list[q][2];
        data->lambda[q][6] = (1.0 - p_list[q][0])*p_list[q][1]*p_list[q][2];
        data->lambda[q][7] = p_list[q][0]*p_list[q][1]*p_list[q][2];
        for (unsigned int i=0; i<vertices_per_cell; ++i)
        {
          for (unsigned int j=0; j<vertices_per_cell; ++j)
          {
            data->sigma_imj_values[q][i][j] = data->sigma[q][i] - data->sigma[q][j];
            
            data->lambda_ipj[q][i][j] = data->lambda[q][i] + data->lambda[q][j];
          }
        }
      }
      // We now want some additional information about sigma_imj_values[q][i][j] = sigma[q][i]-sigma[q][j] 
      // In order to calculate values & derivatives of the shape functions we need to know:
      // - The component the sigma_imj value corresponds to - this varies with i & j.
      // - The gradient of the sigma_imj value
      //   - this depends on the component and the direction of the corresponding edge.
      //   - the direction of the edge is determined by sigma_imj_sign[i][j].
      //
      // Note that not every i,j combination is a valid edge (there are only 12 valid edges in 3D),
      // but we compute them all as it simplifies things.
      
      // store the sign of each component x, y, z in the sigma list.
      // can use this to fill in the sigma_imj_component data.
      const int sigma_comp_signs[GeometryInfo<3>::vertices_per_cell][3]
      ={ {-1, -1, -1}, {1, -1, -1}, {-1, 1, -1}, {1, 1, -1},
         {-1, -1, 1}, {1, -1, 1}, {-1, 1, 1}, {1, 1, 1} };
      
      for (unsigned int i=0; i<vertices_per_cell; ++i)
      {
        for (unsigned int j=0; j<vertices_per_cell; ++j)
        {
          // sigma_imj_sign is the sign (+/-) of the coefficient of x/y/z in sigma_imj.
          // due to the numbering of vertices on the reference element,
          // this is easy to work out since edges in the positive direction
          // go from the smaller to the higher local vertex numbering.
          data->sigma_imj_sign[i][j] = (i < j) ? -1 : 1;
          data->sigma_imj_sign[i][j] = (i == j) ? 0 : data->sigma_imj_sign[i][j];
          
          // Now to store the component which the sigma_i - sigma_j corresponds to:
          for (unsigned int d=0; d<dim; ++d)
          {
            int temp_imj = sigma_comp_signs[i][d] - sigma_comp_signs[j][d];
            // Only interested in the first non-zero
            // as if there is a second, it will not be a valid edge.
            if (temp_imj !=0)
            {
              data->sigma_imj_component[i][j] = d;
              break;
            }
          }
          // Can now calculate the gradient, only non-zero in the component given:
          // Note some i,j combinations will be incorrect, only these will only be on invalid edges.
          data->sigma_imj_grads[i][j][data->sigma_imj_component[i][j]] = 2.0*(double)data->sigma_imj_sign[i][j];
        }
      }     
      // Now compute the edge parameterisations for a single element
      // with global numbering matching that of the reference element:
      
      // resize the edge parameterisations
      data->edge_sigma_values.resize(lines_per_cell);
      data->edge_lambda_values.resize(lines_per_cell);
      data->edge_sigma_grads.resize(lines_per_cell);
      data->edge_lambda_grads_3d.resize(lines_per_cell);
      data->edge_lambda_gradgrads_3d.resize(lines_per_cell);
      for (unsigned int m=0; m<lines_per_cell; ++m)
      {        
        data->edge_sigma_values[m].resize(n_q_points);
        data->edge_lambda_values[m].resize(n_q_points);
        
        // sigma grads are constant in a cell (no need for quad points)
        data->edge_sigma_grads[m].resize(dim);
        
        data->edge_lambda_grads_3d[m].resize(n_q_points);
        for (unsigned int q=0; q<n_q_points; ++q)
        {
          data->edge_lambda_grads_3d[m][q].resize(dim);          
        }
        //lambda_gradgrads are constant in a cell (no need for quad points)
        data->edge_lambda_gradgrads_3d[m].resize(dim);
        for (unsigned int d=0; d<dim; ++d)
        {
          data->edge_lambda_gradgrads_3d[m][d].resize(dim);  
        }
      }
      
      // Fill the values:
      const unsigned int
      edge_sigma_direction[GeometryInfo<3>::lines_per_cell]
      = {1, 1, 0, 0,
        1, 1, 0, 0,
        2, 2, 2, 2};
        
      for (unsigned int m=0; m<lines_per_cell; ++m)
      {
        // e1=max(reference vertex numbering on this edge)
        // e2=min(reference vertex numbering on this edge)
        // Which is guaranteed to be:
        const unsigned int e1 (GeometryInfo<dim>::line_to_cell_vertices(m,1));
        const unsigned int e2 (GeometryInfo<dim>::line_to_cell_vertices(m,0));
        
        for (unsigned int q=0; q<n_q_points; ++q)
        {
          data->edge_sigma_values[m][q] = data->sigma_imj_values[q][e2][e1];
          data->edge_lambda_values[m][q] = data->lambda[q][e1] + data->lambda[q][e2];
        }
        
        data->edge_sigma_grads[m][edge_sigma_direction[m]] = -2.0;
      }
      // lambda grads
      for (unsigned int q=0; q<n_q_points; ++q)
      {
        double x(p_list[q][0]);
        double y(p_list[q][1]);
        double z(p_list[q][2]);
        data->edge_lambda_grads_3d[0][q] = {z-1.0, 0.0, x-1.0};
        data->edge_lambda_grads_3d[1][q] = {1.0-z, 0.0, -x };
        data->edge_lambda_grads_3d[2][q] = {0.0, z-1.0, y-1.0};
        data->edge_lambda_grads_3d[3][q] = {0.0, 1.0-z, -y};
        data->edge_lambda_grads_3d[4][q] = {-z, 0.0, 1.0-x};
        data->edge_lambda_grads_3d[5][q] = {z, 0.0, x};
        data->edge_lambda_grads_3d[6][q] = {0.0, -z, 1.0-y};
        data->edge_lambda_grads_3d[7][q] = {0.0, z, y};
        data->edge_lambda_grads_3d[8][q] = {y-1.0, x-1.0, 0.0};
        data->edge_lambda_grads_3d[9][q] = {1.0-y, -x, 0.0};
        data->edge_lambda_grads_3d[10][q] = {-y, 1.0-x, 0.0};
        data->edge_lambda_grads_3d[11][q] = {y, x, 0.0};
      }
      // edge_lambda gradgrads:
      // TODO: make sure all of these vectors are zero before we begin filling,
      //       otherwise zero entries will have to be set explicitly.
      const int 
      edge_lambda_sign[GeometryInfo<3>::lines_per_cell]
      = {1, -1, 1, -1,
        -1, 1, -1, 1,
        1, -1, -1, 1}; // sign of the 2nd derivative for each edge.
      const unsigned int
      edge_lambda_directions[GeometryInfo<3>::lines_per_cell][2]
      = { {0,2}, {0,2}, {1,2}, {1,2},
      {0,2}, {0,2}, {1,2}, {1,2},
      {0,1}, {0,1}, {0,1}, {0,1} }; // component which edge_lambda[m] depends on.
      for (unsigned int m=0; m<lines_per_cell; ++m)
      {
        data->edge_lambda_gradgrads_3d[m][edge_lambda_directions[m][0]][edge_lambda_directions[m][1]]
        = (double) edge_lambda_sign[m]*1.0;
        data->edge_lambda_gradgrads_3d[m][edge_lambda_directions[m][1]][edge_lambda_directions[m][0]]
        = (double) edge_lambda_sign[m]*1.0;
      }
      // Precomputation for higher order shape functions,
      // and the face parameterisation.
      if (degree > 0)
      {
        // resize required data:        
        data->face_lambda_values.resize(faces_per_cell);
        data->face_lambda_grads.resize(faces_per_cell);
        // for face-based shape functions:
        for (unsigned int m=0; m < faces_per_cell; ++m)
        {
          data->face_lambda_values[m].resize(n_q_points);
          data->face_lambda_grads[m].resize(3);
        }          
        // Fill in the values (these don't change between cells).
        for (unsigned int q=0; q<n_q_points; ++q)
        {
          double x(p_list[q][0]);
          double y(p_list[q][1]);
          double z(p_list[q][2]);
          data->face_lambda_values[0][q] = 1.0-x;
          data->face_lambda_values[1][q] = x;
          data->face_lambda_values[2][q] = 1.0-y;
          data->face_lambda_values[3][q] = y;
          data->face_lambda_values[4][q] = 1.0-z;
          data->face_lambda_values[5][q] = z;
        }
        // gradients are constant:
        data->face_lambda_grads[0] = {-1.0, 0.0, 0.0};
        data->face_lambda_grads[1] = {1.0, 0.0, 0.0};
        data->face_lambda_grads[2] = {0.0, -1.0, 0.0};
        data->face_lambda_grads[3] = {0.0, 1.0, 0.0};
        data->face_lambda_grads[4] = {0.0, 0.0, -1.0};
        data->face_lambda_grads[5] = {0.0, 0.0, 1.0};
        
        // for cell-based shape functions:
        // these don't depend on the cell, so can precompute all here:
        if (flags & update_values | update_gradients)
        {
          // Cell-based shape functions:
          // 
          // Type-1 (gradients):
          // \phi^{C_{1}}_{ijk} = grad( L_{i+2}(2x-1)L_{j+2}(2y-1)L_{k+2}(2z-1) ),
          //
          // 0 <= i,j,k < degree. (in a group of degree*degree*degree)
          const unsigned int cell_type1_offset(n_line_dofs + n_face_dofs);
          // Type-2:
          //
          // \phi^{C_{2}}_{ijk} = diag(1, -1, 1)\phi^{C_{1}}_{ijk}
          // \phi^{C_{2}}_{ijk + p^3} = diag(1, -1, -1)\phi^{C_{1}}_{ijk}
          //
          // 0 <= i,j,k < degree. (subtypes in groups of degree*degree*degree)
          //
          // here we order so that alll of subtype 1 comes first, then subtype 2.      
          const unsigned int cell_type2_offset1(cell_type1_offset + degree*degree*degree);
          const unsigned int cell_type2_offset2(cell_type2_offset1 + degree*degree*degree);
          // Type-3
          // \phi^{C_{3}}_{jk} = L_{j+2}(2y-1)L_{k+2}(2z-1)e_{x}
          // \phi^{C_{3}}_{ik} = L_{i+2}(2x-1)L_{k+2}(2z-1)e_{y}
          // \phi^{C_{3}}_{ij} = L_{i+2}(2x-1)L_{j+2}(2y-1)e_{z}
          //
          // 0 <= i,j,k < degree. (subtypes in groups of degree*degree)
          //
          // again we order so we compute all of subtype 1 first, then subtype 2, etc.
          const unsigned int cell_type3_offset1(cell_type2_offset2 + degree*degree*degree);
          const unsigned int cell_type3_offset2(cell_type3_offset1 + degree*degree);
          const unsigned int cell_type3_offset3(cell_type3_offset2 + degree*degree);
          
          // compute all points we must evaluate the 1d polynomials at:
          std::vector< Point<dim> > cell_points (n_q_points);
          for (unsigned int q=0; q<n_q_points; ++q)
          {
            for (unsigned int d=0; d<dim; ++d)
            {
              cell_points[q][d] = 2.0*p_list[q][d] - 1.0;
            }
          }
          
          
          // only need poly values and 1st derivative for update_values,
          // but need 2nd derivative too for update_gradients.
          const unsigned int poly_length( (flags & update_gradients) ? 3 : 2);
          // TODO: Remove if above works:
          //         if (flags & update_gradients)
          //         {
          //           poly_length=3;
          //         }
          //         else if (flags & update_values)
          //         {
          //           poly_length=2;
          //         }
          std::vector<std::vector<double> > polyx(degree, std::vector<double> (poly_length));
          std::vector<std::vector<double> > polyy(degree, std::vector<double> (poly_length));
          std::vector<std::vector<double> > polyz(degree, std::vector<double> (poly_length));
          // Loop through quad points:
          for (unsigned int q=0; q<n_q_points; ++q)
          {
            // pre-compute values & required derivatives at this quad point:
            // polyx = L_{i+2}(2x-1),
            // polyy = L_{j+2}(2y-1),
            // polyz = L_{k+2}(2z-1).
            //
            // for each polyc[d], c=x,y,z, contains the d-th derivative with respect to the
            // co-ordinate c.
            for (unsigned int i=0; i<degree; ++i)
            {
              // compute all required 1d polynomials for i
              polynomials_1d[i+2].value(cell_points[q][0], polyx[i]);
              polynomials_1d[i+2].value(cell_points[q][1], polyy[i]);
              polynomials_1d[i+2].value(cell_points[q][2], polyz[i]);
            }
            // Now use these to compute the shape functions:
            if (flags & update_values)
            {
              for (unsigned int k=0; k<degree; ++k)
              {
                const unsigned int shift_k(k*degree*degree);
                const unsigned int shift_j(k*degree); // Used below when subbing k for j (type 3)
                for (unsigned int j=0; j<degree; ++j)
                {
                  const unsigned int shift_jk(j*degree + shift_k);
                  for (unsigned int i=0; i<degree; ++i)
                  {
                    const unsigned int shift_ijk(shift_jk + i);
                    
                    // Type 1:
                    const unsigned int dof_index1(cell_type1_offset + shift_ijk);
                    
                    data->shape_values[dof_index1][q][0] = 2.0*polyx[i][1]*polyy[j][0]*polyz[k][0];
                    data->shape_values[dof_index1][q][1] = 2.0*polyx[i][0]*polyy[j][1]*polyz[k][0];
                    data->shape_values[dof_index1][q][2] = 2.0*polyx[i][0]*polyy[j][0]*polyz[k][1];
                    
                    // Type 2:
                    const unsigned int dof_index2_1(cell_type2_offset1 + shift_ijk);
                    const unsigned int dof_index2_2(cell_type2_offset2 + shift_ijk);
                    
                    data->shape_values[dof_index2_1][q][0] = data->shape_values[dof_index1][q][0];
                    data->shape_values[dof_index2_1][q][1] = -1.0*data->shape_values[dof_index1][q][1];
                    data->shape_values[dof_index2_1][q][2] = data->shape_values[dof_index1][q][2];
                    
                    data->shape_values[dof_index2_2][q][0] = data->shape_values[dof_index1][q][0];
                    data->shape_values[dof_index2_2][q][1] = -1.0*data->shape_values[dof_index1][q][1];
                    data->shape_values[dof_index2_2][q][2] = -1.0*data->shape_values[dof_index1][q][2];
                  }
                  // Type 3: (note we re-use k and j for convenience):
                  const unsigned int shift_ij(j+shift_j); // here we've subbed j for i, k for j.
                  const unsigned int dof_index3_1(cell_type3_offset1 + shift_ij);
                  const unsigned int dof_index3_2(cell_type3_offset2 + shift_ij);
                  const unsigned int dof_index3_3(cell_type3_offset3 + shift_ij);
                  
                  data->shape_values[dof_index3_1][q][0] = polyy[j][0]*polyz[k][0];
                  data->shape_values[dof_index3_1][q][1] = 0.0;
                  data->shape_values[dof_index3_1][q][2] = 0.0;
                  
                  data->shape_values[dof_index3_2][q][0] = 0.0;
                  data->shape_values[dof_index3_2][q][1] = polyx[j][0]*polyz[k][0];
                  data->shape_values[dof_index3_2][q][2] = 0.0;
                  
                  data->shape_values[dof_index3_3][q][0] = 0.0;
                  data->shape_values[dof_index3_3][q][1] = 0.0;
                  data->shape_values[dof_index3_3][q][2] = polyx[j][0]*polyy[k][0];
                  
                }
              }
            }
            if (flags & update_gradients)
            {
              for (unsigned int k=0; k<degree; ++k)
              {
                const unsigned int shift_k(k*degree*degree);
                const unsigned int shift_j(k*degree); // Used below when subbing k for j (type 3)
                for (unsigned int j=0; j<degree; ++j)
                {
                  const unsigned int shift_jk(j*degree + shift_k);
                  for (unsigned int i=0; i<degree; ++i)
                  {
                    const unsigned int shift_ijk(shift_jk + i);
                    
                    // Type 1:
                    const unsigned int dof_index1(cell_type1_offset + shift_ijk);
                    
                    data->shape_grads[dof_index1][q][0][0] = 4.0*polyx[i][2]*polyy[j][0]*polyz[k][0];
                    data->shape_grads[dof_index1][q][0][1] = 4.0*polyx[i][1]*polyy[j][1]*polyz[k][0];
                    data->shape_grads[dof_index1][q][0][2] = 4.0*polyx[i][1]*polyy[j][0]*polyz[k][1];
                    
                    data->shape_grads[dof_index1][q][1][0] = data->shape_grads[dof_index1][q][0][1];
                    data->shape_grads[dof_index1][q][1][1] = 4.0*polyx[i][0]*polyy[j][2]*polyz[k][0];
                    data->shape_grads[dof_index1][q][1][2] = 4.0*polyx[i][0]*polyy[j][1]*polyz[k][1];
                    
                    data->shape_grads[dof_index1][q][2][0] = data->shape_grads[dof_index1][q][0][2];
                    data->shape_grads[dof_index1][q][2][1] = data->shape_grads[dof_index1][q][1][2];
                    data->shape_grads[dof_index1][q][2][2] = 4.0*polyx[i][0]*polyy[j][0]*polyz[k][2];
                    
                    // Type 2:
                    const unsigned int dof_index2_1(cell_type2_offset1 + shift_ijk);
                    const unsigned int dof_index2_2(cell_type2_offset2 + shift_ijk);
                    
                    for (unsigned int d=0; d<dim; ++d)
                    {
                      data->shape_grads[dof_index2_1][q][0][d] = data->shape_grads[dof_index1][q][0][d];
                      data->shape_grads[dof_index2_1][q][1][d] = -1.0*data->shape_grads[dof_index1][q][1][d];
                      data->shape_grads[dof_index2_1][q][2][d] = data->shape_grads[dof_index1][q][2][d];
                      
                      data->shape_grads[dof_index2_2][q][0][d] = data->shape_grads[dof_index1][q][0][d];
                      data->shape_grads[dof_index2_2][q][1][d] = -1.0*data->shape_grads[dof_index1][q][1][d];
                      data->shape_grads[dof_index2_2][q][2][d] = -1.0*data->shape_grads[dof_index1][q][2][d];
                    }
                  }
                  // Type 3: (note we re-use k and j for convenience):
                  const unsigned int shift_ij(j+shift_j); // here we've subbed j for i, k for j.
                  const unsigned int dof_index3_1(cell_type3_offset1 + shift_ij);
                  const unsigned int dof_index3_2(cell_type3_offset2 + shift_ij);
                  const unsigned int dof_index3_3(cell_type3_offset3 + shift_ij);
                  // Only need to fill in the non-zero values as the array is initialised to zero.
                  // TODO: CHECK TRUE!!
                  for (unsigned int d1=0; d1<dim; ++d1)
                  {
                    for (unsigned int d2=0; d2<dim; ++d2)
                    {
                      data->shape_grads[dof_index3_1][q][d1][d2] = 0.0;
                      data->shape_grads[dof_index3_2][q][d1][d2] = 0.0;
                      data->shape_grads[dof_index3_3][q][d1][d2] = 0.0;
                    }
                  }
                  data->shape_grads[dof_index3_1][q][0][1] = 2.0*polyy[j][1]*polyz[k][0];
                  data->shape_grads[dof_index3_1][q][0][2] = 2.0*polyy[j][0]*polyz[k][1];
                  
                  data->shape_grads[dof_index3_2][q][1][0] = 2.0*polyx[j][1]*polyz[k][0];
                  data->shape_grads[dof_index3_2][q][1][2] = 2.0*polyx[j][0]*polyz[k][1];
                  
                  data->shape_grads[dof_index3_3][q][2][0] = 2.0*polyx[j][1]*polyy[k][0];
                  data->shape_grads[dof_index3_3][q][2][1] = 2.0*polyx[j][0]*polyy[k][1];
                } 
              }            
            }
          }
        }
      }
      break;
    }
    default:
    {
      Assert (false, ExcNotImplemented ());
    }
  }
  return data;
}

template <int dim>
void MyFE_Nedelec<dim>::fill_edge_values(const typename Triangulation<dim,dim>::cell_iterator &cell,  
                                         const Quadrature<dim>                                &quadrature,
                                         InternalData                                         &fe_data) const
{  
  // This function handles the cell-dependent construction of the EDGE-based shape functions.
  //
  // Note it will handle both 2D and 3D, in 2D, the edges are faces, but we handle them here.
  //
  // It will fill in the missing parts of fe_data which were not possible to fill
  // in the get_data routine, with respect to the edge-based shape functions.
  //
  // It should be called by the fill_fe_*_values routines in order to complete the
  // basis set at quadrature points on the current cell for each edge.
  
  // TODO: Add caching to make this more efficient.
  
  const UpdateFlags flags(fe_data.current_update_flags());
  const unsigned int n_q_points = quadrature.size();
  
  Assert(!(flags & update_values) || fe_data.shape_values.size() == this->dofs_per_cell,
         ExcDimensionMismatch(fe_data.shape_values.size(), this->dofs_per_cell));
  Assert(!(flags & update_values) || fe_data.shape_values[0].size() == n_q_points,
         ExcDimensionMismatch(fe_data.shape_values[0].size(), n_q_points));
  
  // Useful constants:
  const unsigned int degree (this->degree-1); // Note: constructor takes input degree + 1, so need to knock 1 off.

  // Useful geometry info:
  const unsigned int vertices_per_line (2);
  const unsigned int lines_per_cell (GeometryInfo<dim>::lines_per_cell);
  
  const unsigned int dofs_per_line (this->dofs_per_line);
  
  // Calculate edge orderings:
  std::vector<std::vector<unsigned int>> edge_order(lines_per_cell,
                                                    std::vector<unsigned int> (vertices_per_line));
  
  
  switch (dim)
  {
    case 2:
    {
      // TODO:
      //     - Switch the 2D implementation over to match the way 3D handles it
      //     - i.e. calculate values/grads/gradgrads for edge_sigma/lambda in the get_data
      //       routine.
      //     - this streamlines a lot of the following.
      //     - Can then remove the precomputation of a lot of other things.
      
      // Define an edge numbering so that each edge, E_{m} = [e^{m}_{1}, e^{m}_{2}]
      // e1 = higher global numbering of the two local vertices
      // e2 = lower global numbering of the two local vertices
      std::vector<int> edge_sign(lines_per_cell);
      for (unsigned int m=0; m<lines_per_cell; ++m)
      {
        unsigned int v0_loc = GeometryInfo<dim>::line_to_cell_vertices(m,0);
        unsigned int v1_loc = GeometryInfo<dim>::line_to_cell_vertices(m,1);
        unsigned int v0_glob = cell->vertex_index(v0_loc);
        unsigned int v1_glob = cell->vertex_index(v1_loc);
        
        if (v0_glob > v1_glob)
        {
          // Standard as per the deal local edge numberings.
          // NOTE: in 2D we always have non-standard.
          edge_order[m][0] = v0_loc;
          edge_order[m][1] = v1_loc;
          edge_sign[m] = 1.0;
        }
        else
        {
          // Non-standard as per the deal local edge numberings.
          edge_order[m][0] = v1_loc;
          edge_order[m][1] = v0_loc;
          edge_sign[m] = -1.0;
        }
      }
      // TODO: tidy up these comments.. need to reorder things.
      // Define \sigma_{m} = sigma_{e^{m}_{2}} - sigma_{e^{m}_{2}}
      //        \lambda_{m} = \lambda_{e^{m}_{1}} + \lambda_{e^{m}_{2}}
      //
      // To help things, in fe_data, we have precomputed (sigma_{i} - sigma_{j})
      // and (lambda_{i} + lambda_{j}) for 0<= i,j < lines_per_cell.
      //
      // There are two types:
      // - lower order (1 per edge, m):
      //   \phi_{m}^{\mathcal{N}_{0}} = 1/2 grad(\sigma_{m})\lambda_{m}
      //
      // - higher order (degree per edge, m):
      //   \phi_{i}^{E_{m}} = grad( L_{i+2}(\sigma_{m}) (\lambda_{m}) ).
      //
      //   NOTE: sigma_{m} and lambda_{m} are either a function of x OR y
      //         and if sigma is of x, then lambda is of y, and vice versa.
      //         This means that grad(\sigma) requires multiplication by d(sigma)/dx_{i}
      //         for the i^th comp of grad(sigma) and similarly when taking derivatives of lambda.
      //
      // First handle the lowest order edges (dofs 0 to 3)
      // 0 and 1 are the edges in the y dir. (sigma is function of y, lambda is function of x).
      // 2 and 3 are the edges in the x dir. (sigma is function of x, lambda is function of y).
      //
      // More more info: see GeometryInfo for picture of the standard element.
      //
      // Fill edge-based points:
      std::vector<std::vector< Point<dim> > > edge_points(lines_per_cell, std::vector<Point<dim>> (n_q_points));
      std::vector<std::vector< double > > edge_sigma(lines_per_cell, std::vector<double> (n_q_points));
      std::vector<std::vector< double > > edge_lambda(lines_per_cell, std::vector<double> (n_q_points));
      std::vector<std::vector< double > > edge_dsigma(lines_per_cell, std::vector<double> (dim));
      std::vector<std::vector< double > > edge_dlambda(lines_per_cell, std::vector<double> (dim));
      
      // edges 0 and 1
      for (unsigned int m=0; m<2; ++m)
      {    
        const unsigned int
        i (edge_order[m][1]),
        j (edge_order[m][0]);
        
        for (unsigned int q=0; q<n_q_points; ++q)
        {
          edge_sigma[m][q] = fe_data.sigma_imj_values[q][i][j];
          edge_lambda[m][q] = fe_data.lambda_ipj[q][i][j];
          
          edge_points[m][q](0) = edge_lambda[m][q];
          edge_points[m][q](1) = edge_sigma[m][q]; 
        }
        edge_dsigma[m][0] = 0.0;
        edge_dsigma[m][1] = 2.0*edge_sign[m];
        edge_dlambda[m][0] = ((m & 1) == 1 ? 1.0 : -1.0);
        edge_dlambda[m][1] = 0.0;
        
      }  
      // edges 2 and 3
      for (unsigned int m=2; m<4; ++m)
      {    
        const unsigned int
        i (edge_order[m][1]),
        j (edge_order[m][0]);
        
        for (unsigned int q=0; q<n_q_points; ++q)
        {
          edge_sigma[m][q] = fe_data.sigma_imj_values[q][i][j];
          edge_lambda[m][q] = fe_data.lambda_ipj[q][i][j];
          
          edge_points[m][q](0) = edge_sigma[m][q];
          edge_points[m][q](1) = edge_lambda[m][q];
        }
        edge_dsigma[m][0] = 2.0*edge_sign[m];
        edge_dsigma[m][1] = 0.0;
        edge_dlambda[m][0] = 0.0;
        edge_dlambda[m][1] = ((m & 1) == 1 ? 1.0 : -1.0);
        
      }
      
      // TODO: move to fe's data and precompute it
      //       in 2D it'll be (1 1 0 0), 3D needs a little thought. probs 1 1 1 1, 0 0 0 0 and 2 2 2 2 ??
      // NOT USED AT THE MOMENT.. could still be useful.
      std::vector<unsigned int> edge_helper(lines_per_cell);
      edge_helper[0] = 1;
      edge_helper[1] = 1;
      edge_helper[2] = 0;
      edge_helper[3] = 0;
      
      // TODO:
      // Make use of && cell_similarity != CellSimilarity::translation to avoid recomputing things we don't need.
      if (flags & update_values)
      {
        // Lowest order shape functions (edges):
        for (unsigned int q=0; q<n_q_points; ++q)
        {
          for (unsigned int m=0; m<lines_per_cell; ++m)
          {
            // First generate the lowest order functions.
            // These don't require the use of the polynomial space.
            for (unsigned int d=0; d<dim; ++d)
            {
              fe_data.shape_values[0 + m*(dofs_per_line)][q][d] = 0.5*edge_dsigma[m][d]*edge_lambda[m][q];
            }
          }
        }
        if (degree>0)
        {
          // TODO: replace with use of polynomials1d (see grads below too).
          std::vector<double> values (0);
          std::vector<Tensor<1,dim>> grads (polynomial_space.n ());
          std::vector<Tensor<2,dim>> grad_grads (0);
          // Higher order shape functions (edges):
          for (unsigned int q=0; q<n_q_points; ++q)
          {  
            for (unsigned int m=0; m<lines_per_cell; ++m)
            {
              // Compute all gradients at q point.
              polynomial_space.compute(edge_points[m][q], values, grads, grad_grads);
              for (unsigned int dof=1; dof < dofs_per_line; ++dof)
              {
                for (unsigned int d=0; d<dim; ++d)
                {
                  fe_data.shape_values[dof + m*(dofs_per_line)][q][d] = 
                  (edge_dsigma[m][d] + edge_dlambda[m][d])*grads[fe_data.edgeDoF_to_poly[m][dof]][d];
                }
              }
            }
          }
        }
      }
      // TODO: streamline by moving into the same loop as the shape_values
      //       can use similar style to the get_data function.
      if (flags & update_gradients)
      {
        // Lowest order shape functions (edges):
        for (unsigned int q=0; q<n_q_points; ++q)
        {
          for (unsigned int m=0; m<lines_per_cell; ++m)
          {
            // First generate the lowest order functions.
            // These don't require the use of the polynomial space.
            for (unsigned int d1=0; d1<dim; ++d1)
            {
              for (unsigned int d2=0; d2<dim; ++d2)
              {
                fe_data.shape_grads[0 + m*(dofs_per_line)][q][d1][d2]
                = 0.5*edge_dsigma[m][d1]*edge_dlambda[m][d2];
              }
            }
          }
        }
        if (degree>0)
        {
          // TODO: May be more efficient to use polynomials1d (as in the cell-based type 2/3) instead:
          std::vector<double> values (0);
          std::vector<Tensor<1,dim>> grads (0);
          std::vector<Tensor<2,dim>> grad_grads (polynomial_space.n ());
          // Higher order shape functions (edges):
          for (unsigned int q=0; q<n_q_points; ++q)
          {  
            for (unsigned int m=0; m<lines_per_cell; ++m)
            {
              // Compute all gradients of gradients at q point.
              polynomial_space.compute(edge_points[m][q], values, grads, grad_grads);
              for (unsigned int dof=1; dof < degree; ++dof)
              {
                fe_data.shape_grads[dof + m*(dofs_per_line)][q][0][0] =
                edge_dsigma[m][0]*edge_dsigma[m][0]*grad_grads[fe_data.edgeDoF_to_poly[m][dof]][0][0];
                
                double temp = edge_dsigma[m][0]*edge_dlambda[m][1] + edge_dsigma[m][1]*edge_dlambda[m][0];
                fe_data.shape_grads[dof + m*(dofs_per_line)][q][0][1] = temp*grad_grads[fe_data.edgeDoF_to_poly[m][dof]][0][0];
                fe_data.shape_grads[dof + m*(dofs_per_line)][q][1][0] = temp*grad_grads[fe_data.edgeDoF_to_poly[m][dof]][0][0];
                
                fe_data.shape_grads[dof + m*(dofs_per_line)][q][1][1] =
                edge_dsigma[m][1]*edge_dsigma[m][1]*grad_grads[fe_data.edgeDoF_to_poly[m][dof]][0][0];
              }
            }
          }
        }
      }
      break;
    }
    case 3:
    {
      if (flags & update_values | update_gradients)
      {        
        // Define an edge numbering so that each edge, E_{m} = [e^{m}_{1}, e^{m}_{2}]
        // e1 = higher global numbering of the two local vertices
        // e2 = lower global numbering of the two local vertices
        std::vector<int> edge_sign(lines_per_cell);
        for (unsigned int m=0; m<lines_per_cell; ++m)
        {
          unsigned int v0_loc = GeometryInfo<dim>::line_to_cell_vertices(m,0);
          unsigned int v1_loc = GeometryInfo<dim>::line_to_cell_vertices(m,1);
          unsigned int v0_glob = cell->vertex_index(v0_loc);
          unsigned int v1_glob = cell->vertex_index(v1_loc);
          
          if (v0_glob > v1_glob)
          {
            // Opposite to global numbering on our reference element
            edge_sign[m] = -1.0;
          }
          else
          {
            // Aligns with global numbering on our reference element.
            edge_sign[m] = 1.0;
          }
        }
        // TODO: re-order the comments so they flow better.
        //       i.e. move info about the actual shape functions
        //            after the parameterisation of edges, etc.
        //
        // Define \sigma_{m} = sigma_{e^{m}_{2}} - sigma_{e^{m}_{2}}
        //        \lambda_{m} = \lambda_{e^{m}_{1}} + \lambda_{e^{m}_{2}}
        //
        // To help things, in fe_data, we have precomputed (sigma_{i} - sigma_{j})
        // and (lambda_{i} + lambda_{j}) for 0<= i,j < lines_per_cell.
        //
        // There are two types:
        // - lower order (1 per edge, m):
        //   \phi_{m}^{\mathcal{N}_{0}} = 1/2 grad(\sigma_{m})\lambda_{m}
        //
        // - higher order (degree per edge, m):
        //   \phi_{i}^{E_{m}} = grad( L_{i+2}(\sigma_{m}) (\lambda_{m}) ).
        //
        //   NOTE: In the ref cell, sigma_{m} is a function of x OR y OR Z and lambda_{m} a function of the remaining co-ords.
        //         for example, if sigma is of x, then lambda is of y AND z, and so on.
        //         This means that grad(\sigma) requires multiplication by d(sigma)/dx_{i}
        //         for the i^th comp of grad(sigma) and similarly when taking derivatives of lambda.
        //
        // First handle the lowest order edges (dofs 0 to 11)
        // 0 and 1 are the edges in the y dir at z=0. (sigma is a fn of y, lambda is a fn of x & z).
        // 2 and 3 are the edges in the x dir at z=0. (sigma is a fn of x, lambda is a fn of y & z).
        // 4 and 5 are the edges in the y dir at z=1. (sigma is a fn of y, lambda is a fn of x & z).
        // 6 and 7 are the edges in the x dir at z=1. (sigma is a fn of x, lambda is a fn of y & z).
        // 8 and 9 are the edges in the z dir at y=0. (sigma is a fn of z, lambda is a fn of x & y).
        // 10 and 11 are the edges in the z dir at y=1. (sigma is a fn of z, lambda is a fn of x & y).
        //
        // More more info: see GeometryInfo for picture of the standard element.
        //
        // Copy over required edge-based data:
        // TODO: replace this using sigma_imj_values and use signs and components to create the
        //       list of edge_sigma_values on this element.
        std::vector<std::vector< double > > edge_sigma_values(fe_data.edge_sigma_values);
        //       (lines_per_cell,
        //        std::vector<double> (n_q_points));
        std::vector<std::vector< double > > edge_lambda_values(fe_data.edge_lambda_values);
        //       (lines_per_cell,
        //        std::vector<double> (n_q_points));
        
        std::vector<std::vector< double > > edge_sigma_grads(fe_data.edge_sigma_grads);
        //       (lines_per_cell,
        //        std::vector<double> (dim));
        
        std::vector<std::vector<std::vector< double > > > edge_lambda_grads(fe_data.edge_lambda_grads_3d);
        //       (lines_per_cell,
        //        std::vector<std::vector<double> > (n_q_points),
        //        std::Vecotr<double> (dim));
        
        std::vector<std::vector<std::vector< double > > > edge_lambda_gradgrads_3d(fe_data.edge_lambda_gradgrads_3d);
        
        // Adjust the edge_sigma_* for the current cell:
        // TODO: check this works, otherwise add loops?
        for (unsigned int m=0; m<lines_per_cell; ++m)
        {
          std::transform (edge_sigma_values[m].begin(), edge_sigma_values[m].end(),
                          edge_sigma_values[m].begin(),
                          std::bind1st (std::multiplies<double> (),(double) edge_sign[m]));
          std::transform (edge_sigma_grads[m].begin(), edge_sigma_grads[m].end(),
                          edge_sigma_grads[m].begin(),
                          std::bind1st (std::multiplies<double> (), (double) edge_sign[m]));
          //         for (unsigned int q=0; q<n_q_points; 
          //         edge_sigma_values[m] *= edge_sign[m];
          //         edge_sigma_grads[m] *= edge_sign[m];
        }
        
        
        
        // Now calculate the edge-based shape functions:
        // If we want to generate shape gradients then we need second derivatives of the 1d polynomials,
        // but only first derivatives for the shape values.
        const unsigned int poly_length( (flags & update_gradients) ? 3 : 2);
        std::vector<std::vector<double> > poly(degree, std::vector<double> (poly_length));
        for (unsigned int m=0; m<lines_per_cell; ++m)
        {
          const unsigned int shift_m(m*dofs_per_line);
          for (unsigned int q=0; q<n_q_points; ++q)
          {
            // precompute values of all 1d polynomials required:
            // TODO: Could avoid if statement by rearranging to use i=1; i<degree+1
            if (degree>0)
            {
              for (unsigned int i=0; i<degree; ++i)
              {
                polynomials_1d[i+2].value(edge_sigma_values[m][q], poly[i]);
              }              
            }
            if (flags & update_values)
            {
              // Lowest order shape functions:
              // TODO: could probably improve effiency of these loops - sigma_grads doesn't depend on q
              //       and lambda_values doesn't depend on d.
              //       Same goes for the update_gradients part.
              for (unsigned int d=0; d<dim; ++d)
              {
                fe_data.shape_values[shift_m][q][d] = 0.5*edge_sigma_grads[m][d]*edge_lambda_values[m][q];
              }
              // TODO: Could avoid if statement by rearranging to use i=1; i<degree+1
              if (degree>0)
              {
                for (unsigned int i=0; i<degree; ++i)
                {
                  const unsigned int dof_index(i+1 + shift_m);
                  for (unsigned int d=0; d<dim; ++d)
                  {
                    fe_data.shape_values[dof_index][q][d] = edge_sigma_grads[m][d]*poly[i][1]*edge_lambda_values[m][q]
                    + poly[i][0]*edge_lambda_grads[m][q][d];
                  }
                }
              }
            }       
            if (flags & update_gradients)
            {
              // Lowest order shape functions:              
              for (unsigned int d1=0; d1<dim; ++d1)
              {
                for (unsigned int d2=0; d2<dim; ++d2)
                {
                  fe_data.shape_grads[shift_m][q][d1][d2]
                  = 0.5*edge_sigma_grads[m][d1]*edge_lambda_grads[m][q][d2];
                  
                }
              }
              // TODO: Could avoid if statement by rearranging to use i=1; i<degree+1
              if (degree>0)
              {
                for (unsigned int i=0; i<degree; ++i)
                {
                  const unsigned int dof_index(i+1 + shift_m);
                  
                  for (unsigned int d1=0; d1<dim; ++d1)
                  {
                    for (unsigned int d2=0; d2<dim; ++d2)
                    {
                      fe_data.shape_grads[dof_index][q][d1][d2] = edge_sigma_grads[m][d1]*edge_sigma_grads[m][d2]*poly[i][2]*edge_lambda_values[m][q]
                      + ( edge_sigma_grads[m][d1]*edge_lambda_grads[m][q][d2]
                      + edge_sigma_grads[m][d2]*edge_lambda_grads[m][q][d1] )*poly[i][1]
                      + edge_lambda_gradgrads_3d[m][d1][d2]*poly[i][0];
                    }
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
    {
      Assert (false, ExcNotImplemented ());
    }
  }
}

template <int dim>
void MyFE_Nedelec<dim>::fill_face_values(const typename Triangulation<dim,dim>::cell_iterator &cell,
                                         const Quadrature<dim>                                &quadrature,
                                         InternalData                                         &fe_data) const
{
  // This function handles the cell-dependent construction of the FACE-based shape functions.
  //
  // Note that it should only be called in 3D. 
  Assert(dim ==3, ExcDimensionMismatch(dim, 3));  
  //
  // It will fill in the missing parts of fe_data which were not possible to fill
  // in the get_data routine, with respect to face-based shape functions.
  //
  // It should be called by the fill_fe_*_values routines in order to complete the
  // basis set at quadrature points on the current cell for each face.
  
  // TODO: Add caching to make this more efficient.
  
  // Useful constants:
  const unsigned int degree (this->degree-1); // Note: constructor takes input degree + 1, so need to knock 1 off.
  const unsigned int superdegree (this->degree); // Note: constructor takes input degree + 1, so need to knock 1 off.
  
  // Do nothing if FE degree is 0.
  // TODO: could remove, probs not needed.
  if (degree>0)
  {
    const UpdateFlags flags(fe_data.current_update_flags());
    
    if (flags & update_values | update_gradients)
    {
      const unsigned int n_q_points = quadrature.size();
      
      Assert(!(flags & update_values) || fe_data.shape_values.size() == this->dofs_per_cell,
             ExcDimensionMismatch(fe_data.shape_values.size(), this->dofs_per_cell));
      Assert(!(flags & update_values) || fe_data.shape_values[0].size() == n_q_points,
             ExcDimensionMismatch(fe_data.shape_values[0].size(), n_q_points));
      
      // Useful geometry info:
      const unsigned int vertices_per_line (2);
      const unsigned int vertices_per_face (GeometryInfo<dim>::vertices_per_face);
      const unsigned int lines_per_face (GeometryInfo<dim>::lines_per_face);
      const unsigned int lines_per_cell (GeometryInfo<dim>::lines_per_cell);
      const unsigned int faces_per_cell (GeometryInfo<dim>::faces_per_cell);
      
      // DoF info:
      const unsigned int dofs_per_quad (this->dofs_per_quad);
      const unsigned int dofs_per_hex (this->dofs_per_hex);
      
      const unsigned int dofs_per_line (this->dofs_per_line);
      const unsigned int dofs_per_face (this->dofs_per_face);
      const unsigned int dofs_per_cell (this->dofs_per_cell);
      
      const unsigned int n_line_dofs = dofs_per_line*lines_per_cell;
      const unsigned int n_face_dofs = dofs_per_quad*faces_per_cell;
      
      // First we find the global face orientations on the current cell.
      std::vector<std::vector<unsigned int> > face_orientation(faces_per_cell,
                                                               std::vector<unsigned int> (vertices_per_face));
      
      const unsigned int vertex_opposite_on_face[GeometryInfo<3>::vertices_per_face]
      = {3, 2, 1, 0};
      
      const unsigned int vertices_adjacent_on_face[GeometryInfo<3>::vertices_per_face][2]
      = {{1,2}, {0,3}, {0,3}, {1,2}};
      
      for (unsigned int m=0; m<faces_per_cell; ++m)
      {
        // Find the local vertex on this face with the highest global numbering
        // and the face opposite it. This is f^m_0.
        unsigned int current_max = 0;
        unsigned int current_glob = cell->vertex_index(GeometryInfo<dim>::face_to_cell_vertices(m,0));
        for (unsigned int v=1; v<vertices_per_face; ++v)
        {
          if (current_glob < cell->vertex_index(GeometryInfo<dim>::face_to_cell_vertices(m,v)))
          {
            current_max = v;
            current_glob = cell->vertex_index(GeometryInfo<dim>::face_to_cell_vertices(m,v));
          }
        }
        face_orientation[m][0] = GeometryInfo<dim>::face_to_cell_vertices(m,current_max);
        // f^m_2 is the vertex opposite f^m_0:
        face_orientation[m][2] = GeometryInfo<dim>::face_to_cell_vertices(m,vertex_opposite_on_face[current_max]);
        // f^m_1 is the higher global vertex number of the remaining 2 local vertices
        // and f^m_3 is the lower.
        if ( cell->vertex_index(GeometryInfo<dim>::face_to_cell_vertices(m,vertices_adjacent_on_face[current_max][0]))
             > cell->vertex_index(GeometryInfo<dim>::face_to_cell_vertices(m,vertices_adjacent_on_face[current_max][1])) )
        {
          face_orientation[m][1] = GeometryInfo<dim>::face_to_cell_vertices(m,vertices_adjacent_on_face[current_max][0]);
          face_orientation[m][3] = GeometryInfo<dim>::face_to_cell_vertices(m,vertices_adjacent_on_face[current_max][1]);
        }
        else
        {
          face_orientation[m][1] = GeometryInfo<dim>::face_to_cell_vertices(m,vertices_adjacent_on_face[current_max][1]);
          face_orientation[m][3] = GeometryInfo<dim>::face_to_cell_vertices(m,vertices_adjacent_on_face[current_max][0]);
        }
      }
      
      // Now we know the face orientation on the current cell, we have generate the parameterisation:
      std::vector<std::vector<double> > face_xi_values(faces_per_cell,
                                                       std::vector<double> (n_q_points));
      std::vector<std::vector<double> > face_xi_grads(faces_per_cell,
                                                      std::vector<double> (dim));
      std::vector<std::vector<double> > face_eta_values(faces_per_cell,
                                                        std::vector<double> (n_q_points));
      std::vector<std::vector<double> > face_eta_grads(faces_per_cell,
                                                       std::vector<double> (dim));
      
      std::vector<std::vector<double> > face_lambda_values(faces_per_cell,
                                                           std::vector<double> (n_q_points));
      std::vector<std::vector<double> > face_lambda_grads(faces_per_cell,
                                                          std::vector<double> (dim));
      for (unsigned int m=0; m<faces_per_cell; ++m)
      {
        for (unsigned int q=0; q<n_q_points; ++q)
        {
          face_xi_values[m][q] = fe_data.sigma_imj_values[q][face_orientation[m][0]][face_orientation[m][1]];
          face_eta_values[m][q] = fe_data.sigma_imj_values[q][face_orientation[m][0]][face_orientation[m][3]];
          face_lambda_values[m][q] = fe_data.face_lambda_values[m][q];
        }
        for (unsigned int d=0; d<dim; ++d)
        {
          face_xi_grads[m][d] = fe_data.sigma_imj_grads[face_orientation[m][0]][face_orientation[m][1]][d];
          face_eta_grads[m][d] = fe_data.sigma_imj_grads[face_orientation[m][0]][face_orientation[m][3]][d];
          
          face_lambda_grads[m][d] = fe_data.face_lambda_grads[m][d];
        }
      }
      // Now can generate the basis
      const unsigned int poly_length( (flags & update_gradients) ? 3 : 2);
      // TODO: Remove if above works:
      //         if (flags & update_gradients)
      //         {
      //           poly_length=3;
      //         }
      //         else if (flags & update_values)
      //         {
      //           poly_length=2;
      //         }
      std::vector<std::vector<double> > polyxi(degree, std::vector<double> (poly_length));
      std::vector<std::vector<double> > polyeta(degree, std::vector<double> (poly_length));
      
      // Loop through quad points:
      for (unsigned int m=0; m<faces_per_cell; ++m)
      {
        const unsigned int shift_m(m*dofs_per_quad);
        // Calculate the offsets for each face-based shape function:
        // 
        // Type-1 (gradients)
        // \phi^{F_m,1}_{ij} = \nabla( L_{i+2}(\xi_{F_{m}}) L_{j+2}(\eta_{F_{m}}) \lambda_{F_{m}} )
        //
        // 0 <= i,j < degree (in a group of degree*degree)
        const unsigned int face_type1_offset(n_line_dofs + shift_m);
        // Type-2:
        //
        // \phi^{F_m,2}_{ij} = ( L'_{i+2}(\xi_{F_{m}}) L_{j+2}(\eta_{F_{m}}) \nabla\xi_{F_{m}}
        //                       - L_{i+2}(\xi_{F_{m}}) L'_{j+2}(\eta_{F_{m}}) \nabla\eta_{F_{m}} ) \lambda_{F_{m}}
        //
        // 0 <= i,j < degree (in a group of degree*degree)
        const unsigned int face_type2_offset(face_type1_offset + degree*degree);
        // Type-3:
        //
        // \phi^{F_m,3}_{i} = L_{i+2}(\eta_{F_{m}}) \lambda_{F_{m}} \nabla\xi_{F_{m}}
        // \phi^{F_m,3}_{i+p} = L_{i+2}(\xi_{F_{m}}) \lambda_{F_{m}} \nabla\eta_{F_{m}}
        //
        // 0 <= i < degree.
        //
        // here we order so that all of subtype 1 comes first, then subtype 2.
        const unsigned int face_type3_offset1(face_type2_offset + degree*degree);
        const unsigned int face_type3_offset2(face_type3_offset1 + degree);
        
        // Loop over all faces:
        for (unsigned int q=0; q<n_q_points; ++q)
        {          
          // pre-compute values & required derivatives at this quad point:
          // polyxi = L_{i+2}(\xi_{F_{m}}),
          // polyeta = L_{j+2}(\eta_{F_{m}}),
          //
          // each polypoint[k][d], contains the dth derivative of L_{k+2} at the point \xi or \eta.
          // Note that this doesn't include the derivative of xi/eta via the chain rule.
          for (unsigned int i=0; i<degree; ++i)
          {
            // compute all required 1d polynomials:
            polynomials_1d[i+2].value(face_xi_values[m][q], polyxi[i]);
            polynomials_1d[i+2].value(face_eta_values[m][q], polyeta[i]);
          }
          // Now use these to compute the shape functions:
          if (flags & update_values)
          {
            for (unsigned int j=0; j<degree; ++j)
            {
              const unsigned int shift_j(j*degree);
              for (unsigned int i=0; i<degree; ++i)
              {
                const unsigned int shift_ij(shift_j + i);
                // Type 1:
                const unsigned int dof_index1(face_type1_offset + shift_ij);
                for (unsigned int d=0; d<dim; ++d)
                {
                  fe_data.shape_values[dof_index1][q][d] = ( face_xi_grads[m][d]*polyxi[i][1]*polyeta[j][0]
                                                             + face_eta_grads[m][d]*polyxi[i][0]*polyeta[j][1] )*face_lambda_values[m][q]
                                                           + face_lambda_grads[m][d]*polyxi[i][0]*polyeta[j][0];
                }
                // Type 2:
                const unsigned int dof_index2(face_type2_offset + shift_ij);
                for (unsigned int d=0; d<dim; ++d)
                {
                  fe_data.shape_values[dof_index2][q][d] = ( face_xi_grads[m][d]*polyxi[i][1]*polyeta[j][0]
                                                             - face_eta_grads[m][d]*polyxi[i][0]*polyeta[j][1] )*face_lambda_values[m][q];
                }
              }
              // Type 3:
              const unsigned int dof_index3_1(face_type3_offset1 + j);
              const unsigned int dof_index3_2(face_type3_offset2 + j);
              for (unsigned int d=0; d<dim; ++d)
              {
                fe_data.shape_values[dof_index3_1][q][d] = face_xi_grads[m][d]*polyeta[j][0]*face_lambda_values[m][q];

                fe_data.shape_values[dof_index3_2][q][d] = face_eta_grads[m][d]*polyxi[j][0]*face_lambda_values[m][q];
              }
            }
          }
          if (flags & update_gradients)
          {
            for (unsigned int j=0; j<degree; ++j)
            {
              const unsigned int shift_j(j*degree);
              for (unsigned int i=0; i<degree; ++i)
              {
                const unsigned int shift_ij(shift_j + i);
                // Type 1:
                const unsigned int dof_index1(face_type1_offset + shift_ij);
                for (unsigned int d1=0; d1<dim; ++d1)
                {
                  for (unsigned int d2=0; d2<dim; ++d2)
                  {
                    fe_data.shape_grads[dof_index1][q][d1][d2] = ( face_xi_grads[m][d1]*face_xi_grads[m][d2]*polyxi[i][2]*polyeta[j][0]
                                                                   + ( face_xi_grads[m][d1]*face_eta_grads[m][d2]
                                                                       + face_xi_grads[m][d2]*face_eta_grads[m][d1]
                                                                     )*polyxi[i][1]*polyeta[j][1]                                                                       
                                                                       + face_eta_grads[m][d1]*face_eta_grads[m][d2]*polyxi[i][0]*polyeta[j][2]
                                                                 )*face_lambda_values[m][q]
                                                                 + ( face_xi_grads[m][d2]*polyxi[i][1]*polyeta[j][0]
                                                                      + face_eta_grads[m][d2]*polyxi[i][0]*polyeta[j][1]
                                                                   )*face_lambda_grads[m][d1]
                                                                 + ( face_xi_grads[m][d1]*polyxi[i][1]*polyeta[j][0]
                                                                     + face_eta_grads[m][d1]*polyxi[i][0]*polyeta[j][1]
                                                                   )*face_lambda_grads[m][d2];
                  }
                }
                // Type 2:
                const unsigned int dof_index2(face_type2_offset + shift_ij);
                for (unsigned int d1=0; d1<dim; ++d1)
                {
                  for (unsigned int d2=0; d2<dim; ++d2)
                  {
                    fe_data.shape_grads[dof_index2][q][d1][d2] = ( face_xi_grads[m][d1]*face_xi_grads[m][d2]*polyxi[i][2]*polyeta[j][0] 
                                                                   + ( face_xi_grads[m][d1]*face_eta_grads[m][d2]
                                                                       - face_xi_grads[m][d2]*face_eta_grads[m][d1]
                                                                     )*polyxi[i][1]*polyeta[j][1]                                                                       
                                                                   - face_eta_grads[m][d1]*face_eta_grads[m][d2]*polyxi[i][0]*polyeta[j][2]
                                                                 )*face_lambda_values[m][q]
                                                                 + ( face_xi_grads[m][d1]*polyxi[i][1]*polyeta[j][0]
                                                                     - face_eta_grads[m][d1]*polyxi[i][0]*polyeta[j][1]
                                                                   )*face_lambda_grads[m][d2];
                  }
                }
              }
              // Type 3:
              const unsigned int dof_index3_1(face_type3_offset1 + j);
              const unsigned int dof_index3_2(face_type3_offset2 + j);
              for (unsigned int d1=0; d1<dim; ++d1)
              {
                for (unsigned int d2=0; d2<dim; ++d2)
                {
                  fe_data.shape_grads[dof_index3_1][q][d1][d2] = face_xi_grads[m][d1]*( face_eta_grads[m][d2]*polyeta[j][1]*face_lambda_values[m][q]
                                                                                            + face_lambda_grads[m][d2]*polyeta[j][0] );
                                                                     
                  fe_data.shape_grads[dof_index3_2][q][d1][d2] = face_eta_grads[m][d1]*( face_xi_grads[m][d2]*polyxi[j][1]*face_lambda_values[m][q]
                                                                                             + face_lambda_grads[m][d2]*polyxi[j][0] );
                }
              }
            }
          }
        }
      }
    }
    if (flags & update_hessians)
    {
      Assert(false, ExcNotImplemented());
    }
  }
}

template <int dim>
void MyFE_Nedelec<dim>::fill_fe_values(
  const Mapping<dim,dim>                                 &mapping,
  const typename Triangulation<dim,dim>::cell_iterator   &cell,
  const Quadrature<dim>                                  &quadrature,
  typename Mapping<dim,dim>::InternalDataBase            &mapping_data,
  typename Mapping<dim,dim>::InternalDataBase            &fedata,
  FEValuesData<dim,dim>                                  &data,
  CellSimilarity::Similarity                             &cell_similarity) const
{
  
  // Convert to the correct internal data class for this FE class.
  Assert (dynamic_cast<InternalData *> (&fedata) != 0,
          ExcInternalError());
  InternalData &fe_data = static_cast<InternalData &> (fedata);
  
  // Now update the edge-based DoFs, which depend on the cell.
  // This will fill in the missing items in the InternalData (fedata/fe_data)
  // which was not filled in by get_data.
  fill_edge_values(cell, quadrature, fe_data);
  if (dim == 3 && this->degree>1)
  {
    fill_face_values(cell, quadrature, fe_data);
  }
  
  const UpdateFlags flags(fe_data.current_update_flags());
  const unsigned int n_q_points = quadrature.size();
  
  Assert(!(flags & update_values) || fe_data.shape_values.size() == this->dofs_per_cell,
         ExcDimensionMismatch(fe_data.shape_values.size(), this->dofs_per_cell));
  Assert(!(flags & update_values) || fe_data.shape_values[0].size() == n_q_points,
         ExcDimensionMismatch(fe_data.shape_values[0].size(), n_q_points));
  
  const unsigned int dofs_per_cell (this->dofs_per_cell);

  // TODO:
  // Make use of && cell_similarity != CellSimilarity::translation to avoid recomputing things we don't need.
  // May not be possible due to edge reorderings.
  if (flags & update_values)
  {
    // Now have all shape_values stored on the reference cell.
  // Must now transform to the physical cell.
    std::vector<Tensor<1,dim>> transformed_shape_values(n_q_points);
    for (unsigned int dof=0; dof<dofs_per_cell; ++dof)
    {
      const unsigned int first
      = data.shape_function_to_row_table[dof * this->n_components() +
      this->get_nonzero_components(dof).first_selected_component()];
    
      mapping.transform(fe_data.shape_values[dof], transformed_shape_values,
                        mapping_data, mapping_covariant);
      for (unsigned int q=0; q<n_q_points; ++q)
      {
        for (unsigned int d=0; d<dim; ++d)
        {
          data.shape_values(first+d,q) = transformed_shape_values[q][d];
        }
      }
    }
  }
  // TODO: streamline by moving into the same loop as the shape_values
  //       can use similar style to the get_data function.
  if (flags & update_gradients)
  {
    // Now have all shape_grads stored on the reference cell.
    // Must now transform to the physical cell.
    std::vector<Tensor<2,dim>> input(n_q_points);
    std::vector<Tensor<2,dim>> transformed_shape_grads(n_q_points);
    for (unsigned int dof=0; dof<dofs_per_cell; ++dof)
    {
      for (unsigned int q=0; q<n_q_points; ++q)
      {
        input[q] = fe_data.shape_grads[dof][q];
      }                                   
      mapping.transform(input, transformed_shape_grads,
                        mapping_data, mapping_covariant_gradient);

      const unsigned int first
      = data.shape_function_to_row_table[dof * this->n_components() +
                                         this->get_nonzero_components(dof).first_selected_component()];
      for (unsigned int q=0; q<n_q_points; ++q)
      {
        for (unsigned int d=0; d<dim; ++d)
        {
          data.shape_gradients[first+d][q] = transformed_shape_grads[q][d];
        }
      }
    }
  }
}

template <int dim>
void MyFE_Nedelec<dim>::fill_fe_face_values (
  const Mapping<dim,dim>                               &mapping,
  const typename Triangulation<dim,dim>::cell_iterator &cell,
  const unsigned int                                    face,
  const Quadrature<dim-1>                              &quadrature,
  typename Mapping<dim,dim>::InternalDataBase          &mapping_data,
  typename Mapping<dim,dim>::InternalDataBase          &fedata,
  FEValuesData<dim,dim>                                &data) const
{
  // TODO: problem is that we don't have the full quadrature.. should use QProjector to create the 2D quadrature.
  
  // TODO: For now I am effectively generating all of the shape function vals/grads, etc
  //       On all quad points on all faces and then only using them for one face.
  //       This is obviously inefficient. I should cache the cell number and cache
  //       all of the shape_values/gradients etc and then reuse them for each face.
  
  // convert data object to internal
  // data for this class. fails with
  // an exception if that is not
  // possible
  Assert (dynamic_cast<InternalData *> (&fedata) != 0,
          ExcInternalError());
  InternalData &fe_data = static_cast<InternalData &> (fedata);
  
  // Now update the edge-based DoFs, which depend on the cell.
  // This will fill in the missing items in the InternalData (fedata/fe_data)
  // which was not filled in by get_data.
  fill_edge_values(cell,
                   QProjector<dim>::project_to_all_faces(quadrature),
                   fe_data);
  if (dim == 3 && this->degree>1)
  {
    fill_face_values(cell,
                     QProjector<dim>::project_to_all_faces(quadrature),
                     fe_data);
  }
  
  
  
  const UpdateFlags flags(fe_data.update_once | fe_data.update_each);
  const unsigned int n_q_points = quadrature.size();

  // TODO: Make sense of this: should we use the orientatation/flip/rotation information??
  // offset determines which data set
  // to take (all data sets for all
  // faces are stored contiguously)
  const typename QProjector<dim>::DataSetDescriptor offset
    = QProjector<dim>::DataSetDescriptor::face (face, // true, false, false, n_q_points);
                                                cell->face_orientation(face),
                                                cell->face_flip(face),
                                                cell->face_rotation(face),
                                                n_q_points);

  const unsigned int dofs_per_cell (this->dofs_per_cell);

  if (flags & update_values)
  {    
    // Now have all shape_values stored on the reference cell.
    // Must now transform to the physical cell.
    std::vector<Tensor<1,dim>> transformed_shape_values(n_q_points);
    for (unsigned int dof=0; dof<dofs_per_cell; ++dof)
    {
      mapping.transform (make_slice (fe_data.shape_values[dof], offset, n_q_points),
                         transformed_shape_values, mapping_data, mapping_covariant);
      
      const unsigned int first
      = data.shape_function_to_row_table[dof * this->n_components() +
                                         this->get_nonzero_components(dof).first_selected_component()];
      
      for (unsigned int q=0; q<n_q_points; ++q)
      {
        for (unsigned int d=0; d<dim; ++d)
        {
          data.shape_values(first+d,q) = transformed_shape_values[q][d];
        }
      }
    }   
  }
  if (flags & update_gradients)
  {
    // Now have all shape_grads stored on the reference cell.
    // Must now transform to the physical cell.
    std::vector<Tensor<2,dim>> transformed_shape_grads(n_q_points);
    for (unsigned int dof=0; dof<dofs_per_cell; ++dof)
    {
      mapping.transform (make_slice(fe_data.shape_grads[dof], offset, n_q_points),
                         transformed_shape_grads, mapping_data, mapping_covariant_gradient);
      
      const unsigned int first
      = data.shape_function_to_row_table[dof * this->n_components() +
                                         this->get_nonzero_components(dof).first_selected_component()];
      for (unsigned int q=0; q<n_q_points; ++q)
      {
        for (unsigned int d=0; d<dim; ++d)
        {
          data.shape_gradients[first+d][q] = transformed_shape_grads[q][d];
        }
      }
    }
  }
}

template <int dim>
void MyFE_Nedelec<dim>::fill_fe_subface_values(
  const Mapping<dim,dim>                               &mapping,
  const typename Triangulation<dim,dim>::cell_iterator &cell,
  const unsigned int                                    face_no,
  const unsigned int                                    sub_no,
  const Quadrature<dim-1>                              &quadrature,
  typename Mapping<dim,dim>::InternalDataBase          &mapping_data,
  typename Mapping<dim,dim>::InternalDataBase          &fedata,
  FEValuesData<dim,dim>                                &data) const
{
  // TODO: imeplement this.
  Assert (false, ExcNotImplemented ());
}

template <int dim>
UpdateFlags
MyFE_Nedelec<dim>::update_once (const UpdateFlags flags) const
{
  // TODO: Not sure if there is anything we can do here,
  //       have used that from FE_PolyTensor for now.
  const bool values_once = (mapping_type == mapping_none);
  
  UpdateFlags out = update_default;
  if (values_once && (flags & update_values))
    out |= update_values;

  return out;
}

template <int dim>
UpdateFlags
MyFE_Nedelec<dim>::update_each (const UpdateFlags flags) const
{
  UpdateFlags out = update_default;

  if (flags & update_values)
    out |= update_values | update_covariant_transformation;
  
  if (flags & update_gradients)
    out |= update_gradients | update_covariant_transformation;

  if (flags & update_hessians)
  {
//     Assert (false, ExcNotImplemented());
    out |= update_hessians | update_covariant_transformation;
  }

  return out;
}

template <int dim>
std::string
MyFE_Nedelec<dim>::get_name () const
{
  // note that the
  // FETools::get_fe_from_name
  // function depends on the
  // particular format of the string
  // this function returns, so they
  // have to be kept in synch

  std::ostringstream namebuf;
  namebuf << "MyFE_Nedelec<" << dim << ">(" << this->degree-1 << ")";

  return namebuf.str();
}

template <int dim>
FiniteElement<dim>
*MyFE_Nedelec<dim>::clone () const
{
  return new MyFE_Nedelec<dim> (*this);
}


template <int dim>
std::vector<unsigned int> MyFE_Nedelec<dim>::get_dpo_vector (unsigned int degree)
{
  // internal function to return a vector of "dofs per object"
  // where the objects inside the vector refer to:
  // 0 = vertex
  // 1 = edge
  // 2 = face (which is a cell in 2D)
  // 3 = cell 
  std::vector<unsigned int> dpo (dim+1);
  dpo[0] = 0;
  dpo[1] = degree + 1;
  dpo[2] = 2 * degree * (degree + 1);
  if (dim == 3)
  {
    dpo[3] = 3 * degree * degree * (degree + 1);
  }  
  return dpo;
}

template <int dim>
unsigned int MyFE_Nedelec<dim>::compute_n_pols (unsigned int degree)
{
  // Internal function to compute the number of DoFs
  // for a given dimension & polynomial order.
  switch (dim)
  {     
    case 2:
      return 2 * (degree + 1) * (degree + 2);

    case 3:
      return 3 * (degree + 1) * (degree + 2) * (degree + 2);

    default:
    {
      Assert (false, ExcNotImplemented ());
      return 0;
    }  
  }
}

template <int dim>
std::vector<std::vector< Polynomials::Polynomial<double> > >
MyFE_Nedelec<dim>::create_polynomials (const unsigned int degree)
{
  // fill the 1d polynomials vector:
  polynomials_1d = MyPolynomials::integratedLegendre::generate_complete_basis (degree + 1);
  
  std::vector<std::vector< Polynomials::Polynomial<double> > > pols (dim);
  for ( unsigned int i=0; i<dim; ++i)
  {
    pols[i] = polynomials_1d;
  }

  return pols;
}
// Template instantiation
template class MyFE_Nedelec<2>;
template class MyFE_Nedelec<3>;