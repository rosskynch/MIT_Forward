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

#include <forwardsolver.h>

using namespace dealii;

namespace ForwardSolver
{
  // CLASS EDDY CURRENT MEMBERS:
  template <int dim, class DH>
  EddyCurrent<dim, DH>::EddyCurrent(DH &dof_handler,
                                    const FiniteElement<dim> &fe,
                                    const bool direct_flag)
  :
  fe(&fe),
  mapping(&StaticMappingQ1<dim>::mapping),
  direct(direct_flag)
  {
    
    // Constructor for the class:
    // 
    // Input:
    // - DoFHandler with a triangulation and finite element attached (can used dof_handler.initialise).
    // - Finite element, which should be a FESystem containing two blocks/copies of an FE_Nedelec element.
    // 
    // Output:
    // - The modified DoFHandler.
    
    constructor_setup(dof_handler,
                      direct_flag);
  }
  template <int dim, class DH>
  EddyCurrent<dim, DH>::EddyCurrent(const Mapping<dim> &mapping_in,
                                    DH &dof_handler,
                                    const FiniteElement<dim> &fe,
                                    const bool direct_flag)
  :
  fe(&fe),
  mapping(&mapping_in),
  direct(direct_flag)
  {
    constructor_setup(dof_handler,
                      direct_flag);
  }
  
  template<int dim, class DH>
  void EddyCurrent<dim, DH>::constructor_setup(DH &dof_handler,
                                               const bool direct_flag)
                                          
  {
    // Code common to either version of the contructor
    // 
    // This will distribute the degrees of freedom for the input FiniteElement and then re-order
    // the DoF numbering into blocks of lower and higher order basis functions.
    //
    // Finally, the constraints and sparsity pattern for the matrix is created using a dummy boundary function.
    // If the input DoFHandler was attached to a triangulation with non-Dirichlet BCs (boundary_id > 0) then
    // the constraints will only contain hanging node constraints (if present).
    
    // set private direct solver flag:
    direct = direct_flag;

    p_order = fe->degree - 1;
    
    //TODO: work out what the order should be.
    // Some quick tests suggest it should be 2*N+4..
    // this is when the voltages didn't change against the next order up (2*N+5).
    quad_order = 2*(fe->degree) + 1;
    
    // Setup for FEM:
    system_matrix.clear();
    system_preconditioner.clear();
    
    // Renumber by lower/higher order
    
    std::vector<unsigned int> reorder_counts;
    MyDoFRenumbering::my_renumbering<dim, DoFHandler<dim>>(dof_handler, reorder_counts);
    
    // Fill in the DoF count storage:
    n_lowest_order_dofs = reorder_counts[0];
    n_gradient_real = reorder_counts[1];
    n_nongradient_real = reorder_counts[2];
    n_gradient_imag = reorder_counts[3];
    n_nongradient_imag = reorder_counts[4];

    
    n_higher_order_dofs
    = n_gradient_real
    + n_nongradient_real
    + n_gradient_imag
    + n_nongradient_imag;

    n_higher_order_gradient_dofs
    = n_gradient_real
    + n_gradient_imag;
    
    n_higher_order_non_gradient_dofs
    = n_nongradient_real
    + n_nongradient_imag;
    
    end_lowest_order_dofs = n_lowest_order_dofs;
    end_gradient_dofs_real = end_lowest_order_dofs + n_gradient_real;
    end_nongradient_dofs_real = end_gradient_dofs_real + n_nongradient_real;
    end_gradient_dofs_imag = end_nongradient_dofs_real + n_gradient_imag;
    end_nongradient_dofs_imag = end_gradient_dofs_imag + n_nongradient_imag;
    
    // Check that all dofs are accounted for.
    int remaining_dofs = dof_handler.n_dofs()
    - n_lowest_order_dofs
    - n_gradient_real
    - n_nongradient_real
    - n_gradient_imag
    - n_nongradient_imag;
    
    if (remaining_dofs !=0)
    {
      std::cout << std::endl << "WARNING! Renumbering did not find all DoFs! " << remaining_dofs << " were not renumbered." << std::endl;
    }
    
    // Setup initial constraints in order to pre-construct the sparsity pattern:
    constraints.clear ();
    
    
    // TODO: This calling of project_boundary_values_curl_conforming_l2 is very wasteful at this point.
    //       What we actually need to do is simply flag the DoFs which will be constrained by this function.
    //       That is as simple as looking at the boundary_id of edges/faces of each cell and then findings the DoFs.
    //       Then we can just mirror the bits of the project_boundary_values_curl_conforming_l2 which pick out the DoFs
    //       to constrain, but we can ignore the projection part, i.e. just set them as homogenous constraints for now.
    //       Later on, the call to compute_constraints with the actual boundary function will wipe them and fill them
    //       correctly.
    //     
    // FE_Nedelec boundary constraints: note we just a zero function as we have not passed the actual function yet.
    // Real part (begins at 0):
    VectorTools::project_boundary_values_curl_conforming_l2 (dof_handler,
                                                             0,
                                                             ZeroFunction<dim> (dim+dim),
                                                             0,
                                                             constraints,
                                                             *mapping);
    // Imaginary part (begins at dim):
    VectorTools::project_boundary_values_curl_conforming_l2 (dof_handler,
                                                             dim,
                                                             ZeroFunction<dim> (dim+dim),
                                                             0,
                                                             constraints,
                                                             *mapping);
    
    // Add constraints on gradient-based DoFs outside of the conducting region,
    // i.e. in cells with material_id 0.
    // These DoFs are automatically zero in this region because the curl-curl operator
    // applied to a gradient-based DoF is zero and the coefficient of the mass operator
    // is effectively zero.
    // Note: we must avoid constraining DoFs on the suface of the conductor, so first
    // we make a list of DoFs inside the conductor
    if (p_order > 0 && PreconditionerData::constrain_gradients)
    {
      std::vector<bool> dof_in_conductor (dof_handler.n_dofs());
      std::vector<types::global_dof_index> local_dof_indices (fe->dofs_per_cell);
      
      typename DoFHandler<dim>::active_cell_iterator cell, endc;
      endc = dof_handler.end();
      cell = dof_handler.begin_active();
      
      for (;cell!=endc; ++cell)
      {
        // TODO: update with the conductor material id from stored data.
        if (cell->material_id() == 1)
        {
          cell->get_dof_indices (local_dof_indices);
          for (unsigned int i = 0; i<fe->dofs_per_cell; ++i)
          {
            dof_in_conductor[local_dof_indices[i]] = true;
          }
        }
      }
      // Now loop through all cells in the non-conducting region and set gradient DoFs to be zero,
      // on the condition that they are not flagged by dof_in_conductor.
      cell = dof_handler.begin_active();
      for (;cell!=endc; ++cell)
      {
        // TODO: update with the non-conductor material id from stored data.
        if (cell->material_id() != 1)
        {
          cell->get_dof_indices (local_dof_indices);
          for (unsigned int i = 0; i<fe->dofs_per_cell; ++i)
          {
            const unsigned int dof = local_dof_indices[i];
            if ( ( (dof >= end_lowest_order_dofs && dof < end_gradient_dofs_real)
                 || ( dof >= end_nongradient_dofs_real && dof < end_gradient_dofs_imag) )
               && !dof_in_conductor[dof]
               && !constraints.is_constrained(dof) )
            {
              constraints.add_line(dof);
            }
          }
        }
      }
    }
    
    constraints.close ();
    
    
    if (direct)
    {
      BlockDynamicSparsityPattern dsp (1,1);
      dsp.block(0,0).reinit (dof_handler.n_dofs(), dof_handler.n_dofs());
      dsp.collect_sizes();
      
      DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
      
      sparsity_pattern.copy_from(dsp);
      system_matrix.reinit (sparsity_pattern);
      
      solution.reinit(1);
      solution.block(0).reinit (dof_handler.n_dofs());
      solution.collect_sizes();
      
      system_rhs.reinit(1);
      system_rhs.block(0).reinit(dof_handler.n_dofs());
      system_rhs.collect_sizes();
    }
    else if (p_order == 0)
    {
      BlockDynamicSparsityPattern dsp (1,1);
      dsp.block(0,0).reinit (n_lowest_order_dofs, n_lowest_order_dofs);
      dsp.collect_sizes();
      
      DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
      
      sparsity_pattern.copy_from(dsp);
      system_matrix.reinit (sparsity_pattern);
      if (!direct)
      {
        // For lowest order, the preconditioner and system matrix share the same pattern.
        preconditioner_sparsity_pattern.copy_from(dsp);
        system_preconditioner.reinit (preconditioner_sparsity_pattern);
      }
      
      solution.reinit(1);
      solution.block(0).reinit (n_lowest_order_dofs);
      solution.collect_sizes();
      
      system_rhs.reinit(1);
      system_rhs.block(0).reinit(n_lowest_order_dofs);
      system_rhs.collect_sizes();
    }
    else
    {
      // Higher order, setup according to the reordering computed above.
      BlockDynamicSparsityPattern dsp(reorder_counts.size(),reorder_counts.size());
      for (unsigned int i=0; i<reorder_counts.size(); ++i)
      {
        for (unsigned int j=0; j<reorder_counts.size(); ++j)
        {
          dsp.block(i,j).reinit (reorder_counts[i],reorder_counts[j]);
        }
      }
      dsp.collect_sizes();
      
      DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
      
      sparsity_pattern.copy_from(dsp);
      system_matrix.reinit (sparsity_pattern);
      
      if (!direct)
      {
        // For higher orders, we only want the diagonal blocks.
        BlockDynamicSparsityPattern precon_dsp(reorder_counts.size(),reorder_counts.size());
        for (unsigned int i=0; i<reorder_counts.size(); ++i)
        {
          for (unsigned int j=0; j<reorder_counts.size(); ++j)
          {
            precon_dsp.block(i,j).reinit (reorder_counts[i],reorder_counts[j]);
          }
        }
        precon_dsp.collect_sizes();

        DoFTools::make_sparsity_pattern(dof_handler, precon_dsp, constraints, false);
        
        // Now remove the off-diagonals. These are not needed for the preconditioner:
        for (unsigned int i=0; i<reorder_counts.size(); ++i)
        {
          for (unsigned int j=0; j<reorder_counts.size(); ++j)
          {
            if (i!=j)
            {
              precon_dsp.block(i,j).reinit(reorder_counts[i],reorder_counts[j]);
            }
          }
        }
        precon_dsp.collect_sizes();
        
        preconditioner_sparsity_pattern.copy_from(precon_dsp);
        system_preconditioner.reinit (preconditioner_sparsity_pattern);
        
      }
      
      solution.reinit(reorder_counts.size());
      for (unsigned int i=0; i<reorder_counts.size(); ++i)
      {
        solution.block(i).reinit(reorder_counts[i]);
      }
      solution.collect_sizes();
      
      system_rhs.reinit(reorder_counts.size());
      for (unsigned int i=0; i<reorder_counts.size(); ++i)
      {
        system_rhs.block(i).reinit(reorder_counts[i]);
      }
      system_rhs.collect_sizes();
      
    }
  }
  
  template <int dim, class DH>
  EddyCurrent<dim, DH>::~EddyCurrent ()
  {
    // Deconstructor: need to delete the pointers to preconditioners
    // TODO: Could this be better handled with a smart pointer?
    
    if (!direct)
    {
      delete preconditioner;
    }
    
  }
  
  template <int dim, class DH>
  void EddyCurrent<dim, DH>::compute_constraints (const DH &dof_handler,
                                                  const EddyCurrentFunction<dim> &boundary_function)
  {
    // This function will compute the constraints for hanging nodes
    // and the given boundary function.
    // 
    // This function should be called every time the boundary condition has been changed.
    // 
    // The function also assumes that the DoFHandler has been correctly set up.
    // i.e. A triangulation and finite element have been attached and distribute_dofs has been called.
    
    constraints.clear ();

    // FE_Nedelec boundary condition:
    // Real part (begins at 0):
    VectorTools::project_boundary_values_curl_conforming_l2 (dof_handler,
                                                             0,
                                                             boundary_function,
                                                             0,
                                                             constraints,
                                                             *mapping);
    // Imaginary part (begins at dim):
    VectorTools::project_boundary_values_curl_conforming_l2 (dof_handler,
                                                             dim,
                                                             boundary_function,
                                                             0,
                                                             constraints,
                                                             *mapping);
    
    // Add constraints on gradient-based DoFs outside of the conducting region,
    // i.e. in cells with material_id 0.
    // These DoFs are automatically zero in this region because the curl-curl operator
    // applied to a gradient-based DoF is zero and the coefficient of the mass operator
    // is effectively zero.
    // Note: we must avoid constraining DoFs on the suface of the conductor, so first
    // we make a list of DoFs inside the conductor
    if (p_order > 0 && PreconditionerData::constrain_gradients)
    {
      std::vector<bool> dof_in_conductor (dof_handler.n_dofs());
      std::vector<types::global_dof_index> local_dof_indices (fe->dofs_per_cell);
      
      typename DoFHandler<dim>::active_cell_iterator cell, endc;
      endc = dof_handler.end();
      cell = dof_handler.begin_active();
      
      for (;cell!=endc; ++cell)
      {
        // TODO: update with the conductor material id from stored data.
        if (cell->material_id() == 1)
        {
          cell->get_dof_indices (local_dof_indices);
          for (unsigned int i = 0; i<fe->dofs_per_cell; ++i)
          {
            dof_in_conductor[local_dof_indices[i]] = true;
          }
        }
      }
      // Now loop through all cells in the non-conducting region and set gradient DoFs to be zero,
      // on the condition that they are not flagged by dof_in_conductor.
      cell = dof_handler.begin_active();
      for (;cell!=endc; ++cell)
      {
        // TODO: update with the non-conductor material id from stored data.
        if (cell->material_id() == 0)
        {
          cell->get_dof_indices (local_dof_indices);
          for (unsigned int i = 0; i<fe->dofs_per_cell; ++i)
          {
            const unsigned int dof = local_dof_indices[i];
            if ( ( (dof >= end_lowest_order_dofs && dof < end_gradient_dofs_real)
                 || ( dof >= end_nongradient_dofs_real && dof < end_gradient_dofs_imag) )
               && !dof_in_conductor[dof]
               && !constraints.is_constrained(dof) )
            {
              constraints.add_line(dof);
            }
          }
        }
      }
    }

    constraints.close ();
  }
  
  template <int dim, class DH>
  void EddyCurrent<dim, DH>::assemble_matrices (const DH &dof_handler)
  {
    /*
     * Function to assemble the system matrix.
     * 
     * Should really only need to be called once for a given
     * set of material parameters.
     */
    QGauss<dim>  quadrature_formula(quad_order);
    
    const unsigned int n_q_points = quadrature_formula.size();
    
    const unsigned int dofs_per_cell = fe->dofs_per_cell;
    
    FEValues<dim> fe_values (*mapping,
                             *fe, quadrature_formula,
                             update_values    |  update_gradients |
                             update_quadrature_points  |  update_JxW_values);
    
    
    // Extractors to real and imaginary parts
    // TODO: this would make things easier:
    const FEValuesExtractors::Vector E_re(0);
    const FEValuesExtractors::Vector E_im(dim);
    
    std::vector<FEValuesExtractors::Vector> vec(2);
    vec[0] = E_re;
    vec[1] = E_im;
    
    // Local cell storage:
    FullMatrix<double> cell_matrix (dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_preconditioner (dofs_per_cell, dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
    
    // Setup regularisation
    const double kappa_regularised[2][2]
    = { {0.0, -EquationData::param_regularisation},
        {-EquationData::param_regularisation, 0.0} };
    
    const typename DH::active_cell_iterator endc = dof_handler.end();
    typename DH::active_cell_iterator cell;
    
    cell = dof_handler.begin_active();
    
    for (; cell!=endc; ++cell)
    {
      fe_values.reinit (cell);
      cell->get_dof_indices (local_dof_indices);
      
      const double current_mur = EquationData::param_mur(cell->material_id());
      const double mur_inv = 1.0/current_mur;
      const double current_kappa_re = EquationData::param_kappa_re(cell->material_id());
      const double current_kappa_im = EquationData::param_kappa_im(cell->material_id());
      // for preconditioner:
      const double current_kappa_magnitude = sqrt(current_kappa_im*current_kappa_im + current_kappa_re*current_kappa_re);
      
      // Store coefficients of the real/imaginary blocks:
      // NOTE, to make matrix symmetric, we multiply the bottom row of blocks by -1
      const double mur_matrix[2][2] = {
        {mur_inv, 0.0},
        {0.0, -mur_inv} };

      const double kappa_matrix[2][2] = {
        {current_kappa_re, -current_kappa_im},
        {-current_kappa_im, -current_kappa_re} };

      const double kappa_matrix_precon[2][2] = {
        {current_kappa_magnitude, 0.0},
        {0.0, current_kappa_magnitude} };
      
      cell_matrix = 0;
      cell_preconditioner = 0;
      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
        if (!constraints.is_constrained(local_dof_indices[i]))
        {
          const unsigned int block_index_i = fe->system_to_block_index(i).first;
          // Construct local matrix:
          for (unsigned int j=i; j<dofs_per_cell; ++j)
          {
            if (!constraints.is_constrained(local_dof_indices[j]))
            {
              const unsigned int block_index_j = fe->system_to_block_index(j).first;
              double mass_part = 0;
              double curl_part = 0;
              
              const double mu_term = mur_matrix[block_index_i][block_index_j];
              const double precon_kappa_term = kappa_matrix_precon[block_index_i][block_index_j];
              double kappa_term;
              // Are we in the conductor?
              // If no, then check if we're in the lowest order block
              // if yes, then use regularised kappa.
              if (cell->material_id() != 1
                && local_dof_indices[i] < end_lowest_order_dofs
                && local_dof_indices[j] < end_lowest_order_dofs)
              {
                kappa_term = kappa_regularised[block_index_i][block_index_j];
              }
              // Otherwise, either in the conductor or outside of the lowest order block
              // So just use the normal kappa value.
              else
              {
                kappa_term = kappa_matrix[block_index_i][block_index_j];
              }
              if (block_index_i == block_index_j)
              {
                for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
                {
                  curl_part
                  += fe_values[vec[block_index_i]].curl(i, q_point)
                  *fe_values[vec[block_index_j]].curl(j, q_point)
                  *fe_values.JxW(q_point);
                  
                  mass_part
                  += fe_values[vec[block_index_i]].value(i, q_point)
                  *fe_values[vec[block_index_j]].value(j, q_point)
                  *fe_values.JxW(q_point);
                }
                
                // Use skew-symmetry to fill matrices:
                cell_matrix(i,j) = mu_term*curl_part + kappa_term*mass_part;
                cell_matrix(j,i) = cell_matrix(i,j);
                
                if (!direct)
                {
                  // Lowest order block, matches cell matrix
                  // This is a single block, containing real and imaginary.
                  if (local_dof_indices[i] < end_lowest_order_dofs
                    && local_dof_indices[j] < end_lowest_order_dofs)
                  {
                    cell_preconditioner(i,j) = cell_matrix(i,j);
                    cell_preconditioner(j,i) = cell_matrix(j,i);
                  }
                  // Real & imaginary higher order gradient blocks, M(|k|)
                  else if ( (local_dof_indices[i] >= end_lowest_order_dofs
                    && local_dof_indices[j] >= end_lowest_order_dofs
                    && local_dof_indices[i] < end_gradient_dofs_real
                    && local_dof_indices[j] < end_gradient_dofs_real)
                    || (local_dof_indices[i] >= end_nongradient_dofs_real
                    && local_dof_indices[j] >= end_nongradient_dofs_real
                    && local_dof_indices[i] < end_gradient_dofs_imag
                    && local_dof_indices[j] < end_gradient_dofs_imag) )
                  {
                    cell_preconditioner(i,j) = precon_kappa_term*mass_part;
                    cell_preconditioner(j,i) = cell_preconditioner(i,j);
                  }
                  // Real & imaginary higher order non-gradient block, C+M(|k|)
                  // Note we keep coeff of C in the precon positive.
                  else
                  {
                    cell_preconditioner(i,j) = mur_inv*curl_part + precon_kappa_term*mass_part;
                    cell_preconditioner(j,i) = cell_preconditioner(i,j);
                  }
                }
              }
              else // off diagonal - curl-curl operator not part of these blocks..
              {
                for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
                {
                  mass_part
                  += fe_values[vec[block_index_i]].value(i,q_point)
                  *fe_values[vec[block_index_j]].value(j,q_point)
                  *fe_values.JxW(q_point);
                }
                // Use skew-symmetry to fill matrices:
                cell_matrix(i,j) = kappa_term*mass_part;
                cell_matrix(j,i) = cell_matrix(i,j);

                // Only need  to fill preconditoner for the lowest order block
                // for the off-diagonals.
                if (!direct
                  && local_dof_indices[i] < end_lowest_order_dofs
                  && local_dof_indices[j] < end_lowest_order_dofs)
                {
                  cell_preconditioner(i,j) = cell_matrix(i,j);
                  cell_preconditioner(j,i) = cell_matrix(j,i);
                }
              }
            }
          }
        }
      }
      // Distribute & constraint to global matrices:
      // Note: need to apply RHS constraints using the columns of their
      // local matrix in the other routine.
      constraints.distribute_local_to_global(cell_matrix,
                                             local_dof_indices,
                                             system_matrix);
      if (!direct)
      {
        constraints.distribute_local_to_global(cell_preconditioner,
                                               local_dof_indices,
                                               system_preconditioner);
      }
    }
  }
  
  // Assemble the rhs for a zero RHS
  // in the governing equation.
  template <int dim, class DH>
  void EddyCurrent<dim, DH>::assemble_rhs (const DH &dof_handler,
                                           const EddyCurrentFunction<dim> &boundary_function)
  {
    // Function to assemble the RHS for a given boundary_function,
    // which implements the RHS of the equation and
    // Dirichlet & Neumann conditions
    // 
    // It first updates the constraints and then computes the RHS.
    //
    // NOTE: this will completely reset the RHS stored within the class.
    
    // Zero the RHS:
    system_rhs = 0;
    
    compute_constraints(dof_handler,
                        boundary_function);
    
    QGauss<dim>  quadrature_formula(quad_order);
    const unsigned int n_q_points = quadrature_formula.size();
    
    QGauss<dim-1> face_quadrature_formula(quad_order);
    const unsigned int n_face_q_points = face_quadrature_formula.size();
    
    const unsigned int dofs_per_cell = fe->dofs_per_cell;

    // Needed to calc the local matrix for distribute_local_to_global
    // Note: only need the columns of the constrained entries.
    FEValues<dim> fe_values (*mapping,
                             *fe, quadrature_formula,
                             update_values    |  update_gradients |
                             update_quadrature_points  |  update_JxW_values);
    
    FEFaceValues<dim> fe_face_values(*mapping,
                                     *fe, face_quadrature_formula,
                                     update_values | update_quadrature_points |
                                     update_normal_vectors | update_JxW_values);
    
    // Extractors to real and imaginary parts
    const FEValuesExtractors::Vector E_re(0);
    const FEValuesExtractors::Vector E_im(dim);
    
    std::vector<FEValuesExtractors::Vector> vec(2);
    vec[0] = E_re;
    vec[1] = E_im;
    
    // Local cell storage:
    Vector<double> cell_rhs (dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    
    // Setup regularisation
    const double kappa_regularised[2][2]
    = { {0.0, -EquationData::param_regularisation},
        {-EquationData::param_regularisation, 0.0} };
    
    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell)
    {
      cell_rhs = 0.0;
      // Material parameters:
      const double current_mur = EquationData::param_mur(cell->material_id());
      const double mur_inv = 1.0/current_mur;
      const double current_kappa_re = EquationData::param_kappa_re(cell->material_id());
      const double current_kappa_im = EquationData::param_kappa_im(cell->material_id());
      // Store coefficients of the real/imaginary blocks:
      // NOTE, to make matrix symmetric, we multiply the bottom row of blocks by -1
      // RHS coeff:
      const double rhs_coefficient[2] = {
        EquationData::rhs_factor,
        -EquationData::rhs_factor};
      // mu_r coeff (note it's real valued, so zero on off-diag).
      const double mur_matrix[2][2] = {
        {mur_inv, 0.0},
        {0.0, -mur_inv} };
      // kappa coeff
      const double kappa_matrix[2][2] = {
        {current_kappa_re, -current_kappa_im},
        {-current_kappa_im, -current_kappa_re} };
      
      // Update the fe values object.
      // TODO: Could we avoid this in cases with zero RHS and homogenous constraints ???
      fe_values.reinit (cell);
      cell->get_dof_indices (local_dof_indices);
      
      // Loop for non-zero right hand side in equation:
      // Only needed if the RHS is non-zero.
      if (!boundary_function.zero_rhs())
      {
        // RHS Storage:
        std::vector<Vector<double> > rhs_value_list(n_q_points, Vector<double>(fe->n_components()));
        Tensor<1,dim> rhs_value;
        
        boundary_function.rhs_value_list(fe_values.get_quadrature_points(), rhs_value_list, cell->material_id());
        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
          if (!constraints.is_constrained(local_dof_indices[i]))
          {
            const unsigned int block_index_i = fe->system_to_block_index(i).first;
            const unsigned int d_shift = block_index_i*dim;
            double rhs_term = 0;
            
            for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
            {
              for (unsigned int d=0; d<dim; ++d)
              {
                rhs_value[d] = rhs_value_list[q_point](d+d_shift);
              }
              rhs_term += rhs_value
              *fe_values[vec[block_index_i]].value(i,q_point)
              *fe_values.JxW(q_point);
            }
            // Remember the J_{s} term is multiplied by mu0, this
            // is communicated by EquationData::rhs_factor, which
            // has been passed into rhs_coefficient, above.
            cell_rhs(i) = rhs_coefficient[block_index_i]*rhs_term;
          }
        }
      }
      
      // Loop over faces for neumann condition:
      // Only needed if the neumann values are non-zero.
      if (!boundary_function.zero_curl())
      {
        // Neumann storage
        std::vector< Vector<double> > neumann_value_list(n_face_q_points, Vector<double>(fe->n_components()));
        Tensor<1,dim> normal_vector;
        Tensor<1,dim> neumann_value;
        Tensor<1,dim> crossproduct_result;
        
        for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
        {
          if (cell->face(face_number)->at_boundary()
            &&
            (cell->face(face_number)->boundary_id() == 10))
          {
            fe_face_values.reinit (cell, face_number);
            
            boundary_function.curl_value_list(fe_face_values.get_quadrature_points(), neumann_value_list);
            // new
            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
              if (!constraints.is_constrained(local_dof_indices[i]))
              {
                const unsigned int block_index_i = fe->system_to_block_index(i).first;
                const unsigned int d_shift = block_index_i*dim;
                double neumann_term = 0;
                
                const double mu_term = mur_matrix[block_index_i][block_index_i];
                
                for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
                {
                  for (unsigned int d=0; d<dim; ++d)
                  {
                    neumann_value[d] = neumann_value_list[q_point](d+d_shift);
                    normal_vector[d] = fe_face_values.normal_vector(q_point)(d);
                  }
                  cross_product(crossproduct_result, normal_vector, neumann_value);
                  
                  neumann_term -= crossproduct_result
                  *fe_face_values[vec[block_index_i]].value(i,q_point)
                  *fe_face_values.JxW(q_point);
                }
                // Note: mur_matrix factors in the sign of the block.
                cell_rhs(i) += mu_term*neumann_term;
              }
            }
          }
        }
      }
      
      // Create the columns of the local matrix which belong to a constrained
      // DoF. These are all that are needed to apply the constraints to the RHS
      cell_matrix = 0;
      
      // TODO: Add a check to avoid this loop entirely if there are no constraints to apply.
      //       This might mean no inhomogenities, or no hanging node constraints, etc.. need to work out how to be sure of this.
      //       At the moment, eliminating the gradient-based DoFs causes this to compute a lot of columns - are they really needed at all??
      for (unsigned int j=0; j<dofs_per_cell; ++j)
      {
        // Check each column to see if it corresponds to a constrained DoF:
        // TODO: This might be better if we used is_inhomogeneously_constrained.
        //       It may save a significant amount of computation if it does work?
        if ( constraints.is_inhomogeneously_constrained(local_dof_indices[j]) )
        {
          const unsigned int block_index_j = fe->system_to_block_index(j).first;
          // If yes, cycle through all rows to fill the column:
          for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
            const unsigned int block_index_i = fe->system_to_block_index(i).first;
            
            double curl_part = 0;
            double mass_part = 0;
            
            const double mu_term = mur_matrix[block_index_i][block_index_j];
            double kappa_term;
            // Are we in the conductor?
            // If no, then check if we're in the lowest order block
            // if yes, then use regularised kappa.
            if (cell->material_id() != 1
              && local_dof_indices[i] < end_lowest_order_dofs
              && local_dof_indices[j] < end_lowest_order_dofs)
            {
              kappa_term = kappa_regularised[block_index_i][block_index_j];
            }
            // Otherwise, either in the conductor or outside of the lowest order block
            // So just use the normal kappa value.
            else
            {
              kappa_term = kappa_matrix[block_index_i][block_index_j];
            }
            if (block_index_i == block_index_j)
            {
              for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
              {
                curl_part
                += fe_values[vec[block_index_i]].curl(i, q_point)
                *fe_values[vec[block_index_j]].curl(j, q_point)
                *fe_values.JxW(q_point);
                  
                mass_part
                += fe_values[vec[block_index_i]].value(i, q_point)
                *fe_values[vec[block_index_j]].value(j, q_point)
                *fe_values.JxW(q_point);
              }
              cell_matrix(i,j) = mu_term*curl_part + kappa_term*mass_part;
            }
            else // off diagonal - curl-curl operator not needed.
            {
              for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
              {
                mass_part
                += fe_values[vec[block_index_i]].value(i,q_point)
                *fe_values[vec[block_index_j]].value(j,q_point)
                *fe_values.JxW(q_point);
              }
              cell_matrix(i,j) = kappa_term*mass_part;
            }
          }
        }
      }
      // Use the cell matrix constructed for the constrained DoF columns
      // to add the local contribution to the global RHS:
      constraints.distribute_local_to_global(cell_rhs, local_dof_indices, system_rhs, cell_matrix);
    }
  }
  
  template<int dim, class DH>
  void EddyCurrent<dim, DH>::initialise_solver() // removed:(const bool direct_solver_flag)
  {
    // Initialise the direct solver or preconditioner depending on solver choice.
    // Always use the direct solver for p=0. // TODO update documentation here.
    // Once initialised, the boolean variable initialised will be set to true, so that
    // the initialisation won't be repeated.
    
    
    if (initialised)
    {
      // Nothing to do.
      // TODO: add reset_solver() which deletes the preconditioner and resets the direct solver.
      return;
    }
    else
    {
      if (direct) // use direct solver (LU factorisation of whole matrix).
      {
        direct_solve.initialize(system_matrix);
        system_matrix.clear(); // clear as it should no longer be needed.
      }
      // Otherwise:
      // Set up the preconditoner
      // the constructor computes everything required so
      // it can be passed straight to a solve() routine.
      else if (p_order == 0) // use low order preconditioner
      {
        preconditioner = new Preconditioner::EddyCurrentPreconditioner_1x1_lowOrder (system_preconditioner);
        system_preconditioner.clear(); // clear as it should no longer be needed.
      }
      else // p>0 - use low & higher order preconditioner with gradients
      {
        preconditioner = new Preconditioner::EddyCurrentPreconditioner_5x5 (system_preconditioner);
        system_preconditioner.clear(); // clear as it should no longer be needed.
      }
      initialised = true;
    }
  }
  
  template<int dim, class DH>
  void EddyCurrent<dim, DH>::solve (Vector<double> &output_solution, unsigned int &n_iterations)
  {
    // Solve the linear system for the Eddy Current problem.
    // Can call initialise_solver() before this routine, otherwise
    // it will do it on the first call.
    if (~initialised)
    {
      initialise_solver();
    }
    if (direct)
    {
      // Use if switch enabled.
      // Must have called initialise_solver.   
      direct_solve.vmult (solution, system_rhs);
      constraints.distribute (solution);
      
      n_iterations = 1;
    }
    else // p>0 use GMRES with low & higher order block preconditioner
    {        
      /*GMRES*/        
      SolverControl solver_control (system_matrix.m(),
                                    PreconditionerData::solver_tolerance*system_rhs.l2_norm(),
                                    true, true); // Add to see residual history
      
      GrowingVectorMemory<BlockVector<double> > vector_memory;
      SolverGMRES<BlockVector<double> >::AdditionalData gmres_data;
      gmres_data.max_n_tmp_vectors = 250;
      gmres_data.right_preconditioning = PreconditionerData::right_preconditioning;
      
      // Care needs to be taken with the residual monitor.
      gmres_data.use_default_residual = PreconditionerData::right_preconditioning ? true : false;

      gmres_data.force_re_orthogonalization = false;
      
      SolverGMRES<BlockVector<double> > gmres(solver_control, vector_memory,
                                              gmres_data);
      
      gmres.solve(system_matrix,
                  solution,
                  system_rhs,
                  *preconditioner);
      
      // Output iterations to screen
      deallog << "GMRES Iterations:                "
      << solver_control.last_step()
      << std::endl;
      
      n_iterations = solver_control.last_step();
      
      constraints.distribute (solution);
    }
  
    // TODO: may not be able to pass to the output solution like this:
    // seems to be ok. for now.
    output_solution.reinit(solution.size());
    output_solution = solution;
  }
  // Template instantiation
  template class EddyCurrent<3, DoFHandler<3>>;
  // END CLASS EDDYCURRENT
}
// END namespace ForwardSolver
