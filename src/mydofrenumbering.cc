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

#include <mydofrenumbering.h>

namespace MyDoFRenumbering
{
  using namespace dealii;
  namespace internals
  {
    // internal function to do the work:
    template <int dim, class DH>
    types::global_dof_index
    compute_by_dimension (std::vector<types::global_dof_index> &renumbering,
                          const DH &dof_handler,
                          std::vector<unsigned int> &dof_counts)
    {
      // Splits DoFs for an FENedelec<dim> into lower order edges and higher order edges/faces/cells
      // 
      // For each copy of each base element, we know there are 12 lowest order
      // edge DoFs on the reference element
      //  
      // Loop through each base element and each multiplicity and grab the system index of each.
      // Can then compare this index to where we know the lowest order DoF is and reorder. 
      
      // TODO: For now this is only designed to work in 3D, so throw an exception in 2D:
      Assert (dim == 3, ExcNotImplemented ());
      
      const FiniteElement<dim> &fe = dof_handler.get_fe ();
      const unsigned int superdegree = fe.degree;
      const unsigned int degree = superdegree - 1;
      const unsigned int n_global_dofs = dof_handler.n_dofs();
      const unsigned int lines_per_cell = GeometryInfo<dim>::lines_per_cell;
      const unsigned int faces_per_cell = GeometryInfo<dim>::faces_per_cell;
      
      const unsigned int dofs_per_line = fe.dofs_per_line;
      const unsigned int dofs_per_quad = fe.dofs_per_quad;
      const unsigned int dofs_per_hex = fe.dofs_per_hex;
      
      const unsigned int dofs_per_face = fe.dofs_per_face; // this includes dofs on lines and faces together.
      const unsigned int dofs_per_cell = fe.dofs_per_cell;
      
      const unsigned int line_dofs_per_cell = lines_per_cell*dofs_per_line;
      const unsigned int face_dofs_per_cell = faces_per_cell*dofs_per_quad;
      
      std::vector<bool> global_dofs_seen (n_global_dofs);
      
      /*
       * Store a temporary array to store all renumberings
       * as well as an array containing a count of how many DoFs have
       * been reordered within each
       * we'll then combine these lists to create the final cell_reordering
       * with a loop over edges, faces and then cells, with the lowest order
       * being first within each.
       */
      
      
      
      for (unsigned int i = 0; i < n_global_dofs; ++i)
      {
        global_dofs_seen[i] = false;
      }
      
      std::pair<unsigned int, unsigned int> base_indices (0,0);
      std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
      /*
       * Idea is:
       * We know that for an FESystem of Nedelec elements we have a local ordering
       * based on 2 (in 2D) or 3 (in 3D) loops:
       * loop 1:
       * base elements
       * -> element copy
       *    -> edges
       *       -> basis polynomial orders
       * 
       * loop 2:
       * base elements
       * -> element copy
       *    -> faces
       *       -> basis polynomial orders
       * 
       * loop 3:
       * base elements
       * -> element copy
       *    -> cells
       *       -> basis polynomial orders
       * 
       * So if we want to group the DoFs by base element, element copy and polynomial order on edges/faces/cells
       * then we can exploit this.
       * 
       * We also know that on a local cell, the DoFs are grouped by:
       * edges then
       * faces then
       * cells
       * 
       * For now we only want to find the lowest order edge DoFs.
       * 
       * within a given copy of a given base element (i.e. we can say base_indices = (base_element, element_copy))
       * we can look at the local cell index, restricting ourselves to edge DoFs since we know they come before faces/cells,
       * (via base_cell_index = fe.system_to_base_index(local_dof_index).second)
       * and know that a lowest order edge function satifies mod(base_cell_index,fe.degree) = 0
       * 
       * similarly, the 1st order (assuming fe.degree>1) edge function satifies
       * mod(base_cell_index,fe.degree) = 1, and so on.
       * 
       * Note: It is not quite so simple for face and cell DoFs, since the number of them as the degree increases
       *       does not scale linearly.
       * 
       */
      
      // Control for the number and types of block:
      // Currently we have:
      //       lower order
      //       gradient-based (edge/face/cell subtypes)
      //       non-gradient-based (face/cell subtypes) (NOTE: lack of edge-based non-gradients!).
      //
      // TODO: could provide an option to re-order this so we have:
      //       lower order
      //       edges (gradients subtype ONLY)
      //       faces (gradients/non-gradients subtypes)
      //       cells (gradients/non-gradients subtypes)
      const unsigned int n_blocks = 6;
      const unsigned int lower_order_block = 0;
      const unsigned int higher_order_edge_gradient_block = 1;
      const unsigned int higher_order_face_gradient_block = 2;
      const unsigned int higher_order_cell_gradient_block = 3;
      const unsigned int higher_order_face_nongradient_block = 4;
      const unsigned int higher_order_cell_nongradient_block = 5;
      
      // Max local edge/face/cell numbering of a gradient-based DoF
      // NOTE: for each edge/face/cell the gradient-based DoFs come first
      // then the non-gradients (see new FE_Nedelec element).
      // There are no non-gradients on edges, so we only need face & cell limits:
      // TODO: this would change in 2D.
      const unsigned int n_gradients_per_face(degree*degree);
      const unsigned int n_gradients_per_cell(degree*degree*degree);
      
      std::vector<std::vector<types::global_dof_index> > all_renumberings(n_blocks,
                                                                         std::vector<types::global_dof_index> (n_global_dofs));
      
      std::vector<std::vector<bool> > all_reordered_dofs(n_blocks,
                                              std::vector<bool> (n_global_dofs));
      
      std::vector<types::global_dof_index> all_dof_counts(n_blocks);
      
      for (unsigned int b=0; b<n_blocks; ++b)
      {
        all_dof_counts[b] = 0;
        for (unsigned int dof=0; dof<n_global_dofs; ++dof)
        {
          all_reordered_dofs[b][dof] = false;
        }
      }
      
      for (unsigned int base_element = 0; base_element < fe.n_base_elements (); ++base_element)
      {
        base_indices.first = base_element;
        for (unsigned int element_copy = 0; element_copy < fe.element_multiplicity (base_element); ++element_copy)
        {
          base_indices.second = element_copy;
          typename DoFHandler<dim>::active_cell_iterator
          cell = dof_handler.begin_active(),
          endc = dof_handler.end();
          for (; cell!=endc; ++cell)
          {
            cell->get_dof_indices (local_dof_indices);
            std::vector<bool> local_dofs_seen (dofs_per_cell);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              local_dofs_seen[i] = false;
            }
            /* 
             * Loop over all edge DoFs.
             * Note that edge dofs are all grouped together
             * so we can separate the lowest orders from
             * the higher orders and not worry about face/cell DoFs.
             */
            for (unsigned int local_dof = 0; local_dof < line_dofs_per_cell; ++local_dof)
            {
              // Lowest order part:
              if ( (fe.system_to_base_index(local_dof).first == base_indices)
                && (fe.system_to_base_index(local_dof).second % fe.base_element(base_element).dofs_per_line == 0)
                && ( !local_dofs_seen[local_dof] )
                && ( !global_dofs_seen[local_dof_indices[local_dof]] ) )
              {
                local_dofs_seen[local_dof] = true;
                global_dofs_seen[local_dof_indices[local_dof]] = true;
                all_renumberings[lower_order_block][local_dof_indices[local_dof]]
                 = all_dof_counts[lower_order_block];
                all_reordered_dofs[lower_order_block][local_dof_indices[local_dof]] = true;
                ++all_dof_counts[lower_order_block];
              }
              else if ( (fe.system_to_base_index(local_dof).first == base_indices)
                && (fe.system_to_base_index(local_dof).second % fe.base_element(base_element).dofs_per_line > 0)
                && ( !local_dofs_seen[local_dof] )
                && ( !global_dofs_seen[local_dof_indices[local_dof]] ) )
              {
                local_dofs_seen[local_dof] = true;
                global_dofs_seen[local_dof_indices[local_dof]] = true;
                // Gradient-based edges. NOTE: There are no non-gradient edge-based DoFs.
                all_renumberings[higher_order_edge_gradient_block][local_dof_indices[local_dof]]
                 = all_dof_counts[higher_order_edge_gradient_block];
                all_reordered_dofs[higher_order_edge_gradient_block][local_dof_indices[local_dof]] = true;
                ++all_dof_counts[higher_order_edge_gradient_block];
              }
            }
            // Loop over all face DoFs, skipping the edge DoFs, stopping
            // before we reach the interior cell DoFs.
            // The ordering of these local dofs is by face, then gradients first, followed by non-gradients.
            for (unsigned int local_dof = line_dofs_per_cell;
                 local_dof < line_dofs_per_cell + face_dofs_per_cell; ++local_dof)
            {
              if ( (fe.system_to_base_index(local_dof).first == base_indices)
                && ( !local_dofs_seen[local_dof] )
                && ( !global_dofs_seen[local_dof_indices[local_dof]] ) )
              {
                local_dofs_seen[local_dof] = true;
                global_dofs_seen[local_dof_indices[local_dof]] = true;
                // To decide if we have a gradient or non-gradient can use
                // the base element's local dof mod dofs_per_face. (after being adjusted to ignore lines).
                // Since gradients come first on each face:
                // If this is between 0 and n_gradients_per_face => gradient,
                //                                     otherwise => non-gradient.
                if ( (fe.system_to_base_index(local_dof).second - lines_per_cell*fe.base_element(base_element).dofs_per_line)
                     % fe.base_element(base_element).dofs_per_quad < n_gradients_per_face )
                {
                  all_renumberings[higher_order_face_gradient_block][local_dof_indices[local_dof]]
                   = all_dof_counts[higher_order_face_gradient_block];
                  all_reordered_dofs[higher_order_face_gradient_block][local_dof_indices[local_dof]] = true;
                  ++all_dof_counts[higher_order_face_gradient_block];
                }
                else
                {
                  all_renumberings[higher_order_face_nongradient_block][local_dof_indices[local_dof]]
                   = all_dof_counts[higher_order_face_nongradient_block];
                  all_reordered_dofs[higher_order_face_nongradient_block][local_dof_indices[local_dof]] = true;
                  ++all_dof_counts[higher_order_face_nongradient_block];
                }
              }
            }
            // Loop over all cell DoFs, skipping the edge and face DoFs,
            // stopping at the last local DoF
            // The ordering of these local dofs is simply by gradients followed by non-gradients.
            for (unsigned int local_dof = line_dofs_per_cell + face_dofs_per_cell;
                 local_dof < dofs_per_cell; ++local_dof)
            {
              if ( (fe.system_to_base_index(local_dof).first == base_indices)
                && ( !local_dofs_seen[local_dof] )
                && ( !global_dofs_seen[local_dof_indices[local_dof]] ) )
              {
                local_dofs_seen[local_dof] = true;
                global_dofs_seen[local_dof_indices[local_dof]] = true;
                // To decide if we have a gradient or non-gradient, just check if the
                // base element's local dof is less than n_gradients_per_cell.
                // (after being shifted to only consider cell interiors).
                if ( (fe.system_to_base_index(local_dof).second
                      - lines_per_cell*fe.base_element(base_element).dofs_per_line
                      - faces_per_cell*fe.base_element(base_element).dofs_per_quad) < n_gradients_per_cell )
                {
                  all_renumberings[higher_order_cell_gradient_block][local_dof_indices[local_dof]]
                   = all_dof_counts[higher_order_cell_gradient_block];
                  all_reordered_dofs[higher_order_cell_gradient_block][local_dof_indices[local_dof]] = true;
                  ++all_dof_counts[higher_order_cell_gradient_block];
                }
                else
                {
                  all_renumberings[higher_order_cell_nongradient_block][local_dof_indices[local_dof]]
                   = all_dof_counts[higher_order_cell_nongradient_block];
                  all_reordered_dofs[higher_order_cell_nongradient_block][local_dof_indices[local_dof]] = true;
                  ++all_dof_counts[higher_order_cell_nongradient_block];
                }
              }
            }
          }
        }
      }
      // Fill the array with counts of each type of DoF
      dof_counts.resize(n_blocks);
      for (unsigned int b=0; b<n_blocks; ++b)
      {
        dof_counts[b] = all_dof_counts[b];
      }

      // Now have reordering for lowest and higher order (split into gradients/non-gradients
      // with edge/face/cell subtypes) dofs.
      // Now want to combine them into a single reordering
      for (unsigned int dof = 0; dof < n_global_dofs; ++dof)
      {
        // Need to keep a track of the dof counts in blocks we've skipped
        // and adjust the renumbering accordingly.
        unsigned int block_increment = 0;
        for (unsigned int b=0; b<n_blocks; ++b)
        {
          if (all_reordered_dofs[b][dof])
          {
            renumbering[dof] = all_renumberings[b][dof] + block_increment;
            break;
          }
          block_increment += all_dof_counts[b];
        }
      }
      /* Check if we've now reordered all DoFs */
      unsigned int total_counted_dofs=0;
      for (unsigned int b=0; b<n_blocks; ++b)
      {
        total_counted_dofs += all_dof_counts[b];
      }

      if (total_counted_dofs == n_global_dofs)
      {
        return total_counted_dofs;
      }
      // If not, then add remaining DoFs in the order they come:
      std::cout << " DoFRenumbering::by_dimension: some DoFs still remain..." << std::endl;
      for (unsigned int dof = 0; dof < n_global_dofs; ++dof)
      {
        if (!global_dofs_seen[dof])
        {
          renumbering[dof] = total_counted_dofs;
          ++total_counted_dofs;
          global_dofs_seen[dof] = true;
        }
      }
      return total_counted_dofs;
    }
  }
  // Short function to call the internal function:
  template <int dim, class DH>
  void by_dimension (DH &dof_handler,
                     std::vector<unsigned int> &dof_counts)
  {
    std::vector<types::global_dof_index> renumbering(dof_handler.n_dofs());
    
    types::global_dof_index result = internals::
                                     compute_by_dimension<dim, DH>(renumbering,
                                                                   dof_handler,
                                                                   dof_counts);
    if (result == 0)
    {
      return;
    }
    dof_handler.renumber_dofs (renumbering);
  }
  
  template
  void by_dimension<3>(DoFHandler<3> &,
                      std::vector<unsigned int> &);
  
  
  // New renumbering, designed for the eddy current problem with the zaglmayr nedelec element:
  // TODO:
  // This could be handled using 2 reorderings, making this code more reusable for other purposes.
  // reordering 1 could produce a set of reorderings for each base element & multiplicity:
  // - by lower order, gradient-based and non-gradient
  // 
  // e.g if there is one base element with multiplicity 2 (or 2 base elements with multiplicity 1 each)
  // then you'd get 6 blocks, the first 3 splitting the first multiplicity into lower/gradient/non-gradient
  //                          the second 3 splitting the second multiplicity into lower/gradient/non-gradient
  //
  // This could be made general by returning counts like this dof_counts[bm][s]
  // where bm= 0,.., sum(base_element.multiplicity) and s=0,..,2 (0=n lower order dofs,  1 = n gradient dofs, 2 = n non-gradient dofs).
  //
  // This could be easily done by creating a vector of base_indices pairs, which correspond to each base element/copy combination.
  // Then for each, just loop through the local DoFs and gather the lower/gradient/non-gradients (could even use edges/faces/cells too) in
  // the reorder counts.
  //
  // Once this is computed, a reordering to group the (for example) lower order dofs together could be used.
  // i.e. something which will group all a particular s with all bm (as labelled above).
  //
  //As time is short, I am just writing this in 1 go.
  namespace internals
  {
    template <int dim, class DH>
    types::global_dof_index
    compute_my_renumbering (std::vector<types::global_dof_index> &renumbering,
                            const DH &dof_handler,
                            std::vector<unsigned int> &dof_counts)
    {
      // Splits DoFs for an FENedelec<dim> into lower order edges and higher order edges/faces/cells
      // 
      // For each copy of each base element, we know there are 12 lowest order
      // edge DoFs on the reference element
      //  
      // Loop through each base element and each multiplicity and grab the system index of each.
      // Can then compare this index to where we know the lowest order DoF is and reorder. 
      
      // For now this is only designed to work in 3D, so throw an exception in 2D:
      Assert (dim == 3, ExcNotImplemented ());
      
      const FiniteElement<dim> &fe = dof_handler.get_fe ();
      const unsigned int superdegree = fe.degree;
      const unsigned int degree = superdegree - 1;
      const unsigned int n_global_dofs = dof_handler.n_dofs();
      const unsigned int lines_per_cell = GeometryInfo<dim>::lines_per_cell;
      const unsigned int faces_per_cell = GeometryInfo<dim>::faces_per_cell;
      
      const unsigned int dofs_per_line = fe.dofs_per_line;
      const unsigned int dofs_per_quad = fe.dofs_per_quad;
      
      const unsigned int dofs_per_cell = fe.dofs_per_cell;
      
      const unsigned int line_dofs_per_cell = lines_per_cell*dofs_per_line;
      const unsigned int face_dofs_per_cell = faces_per_cell*dofs_per_quad;
      
      std::vector<bool> global_dofs_seen (n_global_dofs);
      for (unsigned int i = 0; i < n_global_dofs; ++i)
      {
        global_dofs_seen[i] = false;
      }
      
      std::pair<unsigned int, unsigned int> base_indices (0,0);
      std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
      // Idea is:
      // We know that for an FESystem of Nedelec elements we have a local ordering
      // based on 2 (in 2D) or 3 (in 3D) loops:
      // loop 1:
      // base elements
      // -> element copy
      //    -> edges
      //       -> basis polynomial orders
      // 
      // loop 2:
      // base elements
      // -> element copy
      //    -> faces
      //       -> basis polynomial orders
      // 
      // loop 3:
      // base elements
      // -> element copy
      //    -> cells
      //       -> basis polynomial orders
      // 
      // Now we want to rearrange so that we get the following (note block0 = real part, block1 = imag part).
      // - ALL lower order edges (i.e. all lower orders for both blocks, can have block0 first then block1)
      // - block0 gradients
      // - block0 non-gradients
      // - block1 gradients
      // - block1 non-gradients.
      //
      // I.e. we have 5 blocks in total:
      const unsigned int n_blocks = 5;
      const unsigned int lower_order_block = 0;
      
      // Store the base indicies for block1, must be copy 0 of element 0.
      std::pair<unsigned int, unsigned int> block0_base_indices (0,0);
      const unsigned int gradient_block0 = 1;
      const unsigned int nongradient_block0 = 2;
      
      // Store the base indicies for block1:
      // it must be last copy of last element i.e. could be copy 1 of element 0, or copy 0 of element 1.
      std::pair<unsigned int, unsigned int> block1_base_indices
      (fe.n_base_elements ()-1, fe.element_multiplicity (fe.n_base_elements ()-1)-1);
      const unsigned int gradient_block1 = 3;
      const unsigned int nongradient_block1 = 4;
     
      // Max local edge/face/cell numbering of a gradient-based DoF
      // NOTE: for each edge/face/cell the gradient-based DoFs come first
      // then the non-gradients (see new FE_Nedelec element).
      // There are no non-gradients on edges, so we only need face & cell limits:
      // TODO: this would change in 2D.
      const unsigned int n_gradients_per_face(degree*degree);
      const unsigned int n_gradients_per_cell(degree*degree*degree);
      
      std::vector<std::vector<types::global_dof_index> > all_renumberings(n_blocks,
                                                                         std::vector<types::global_dof_index> (n_global_dofs));
      
      std::vector<std::vector<bool> > all_reordered_dofs(n_blocks,
                                              std::vector<bool> (n_global_dofs));
      
      std::vector<types::global_dof_index> all_dof_counts(n_blocks);
      
      for (unsigned int b=0; b<n_blocks; ++b)
      {
        all_dof_counts[b] = 0;
        for (unsigned int dof=0; dof<n_global_dofs; ++dof)
        {
          all_reordered_dofs[b][dof] = false;
        }
      }
      
      for (unsigned int base_element = 0; base_element < fe.n_base_elements (); ++base_element)
      {
        // Use this to keep lower orders sub-grouped by block0, block1. Although they all form a single block in the
        // preconditioner and we don't keep a careful count of it.
        base_indices.first = base_element;
        for (unsigned int element_copy = 0; element_copy < fe.element_multiplicity (base_element); ++element_copy)
        {
          base_indices.second = element_copy;
          typename DoFHandler<dim>::active_cell_iterator
          cell = dof_handler.begin_active(),
          endc = dof_handler.end();
          for (; cell!=endc; ++cell)
          {
            cell->get_dof_indices (local_dof_indices);
            std::vector<bool> local_dofs_seen (dofs_per_cell);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              local_dofs_seen[i] = false;
            }

            // Loop over all edge DoFs.
            // Note that edge dofs are all grouped together
            // so we can separate the lowest orders from
            // the higher orders and not worry about face/cell DoFs.

            for (unsigned int local_dof = 0; local_dof < line_dofs_per_cell; ++local_dof)
            {
              // Find lowest order dofs:
              if ( (fe.system_to_base_index(local_dof).first == base_indices)
                && (fe.system_to_base_index(local_dof).second % fe.base_element(base_element).dofs_per_line == 0)
                && ( !local_dofs_seen[local_dof] )
                && ( !global_dofs_seen[local_dof_indices[local_dof]] ) )
              {
                local_dofs_seen[local_dof] = true;
                global_dofs_seen[local_dof_indices[local_dof]] = true;
                all_renumberings[lower_order_block][local_dof_indices[local_dof]]
                 = all_dof_counts[lower_order_block];
                all_reordered_dofs[lower_order_block][local_dof_indices[local_dof]] = true;
                ++all_dof_counts[lower_order_block];
              }
              // Otherwise we have a higher order edge, which MUST be a gradient.
              // Now we also need to separate by base element/multiplicity.
              else if ( (fe.system_to_base_index(local_dof).first == block0_base_indices)
                && (fe.system_to_base_index(local_dof).second % fe.base_element(base_element).dofs_per_line > 0)
                && ( !local_dofs_seen[local_dof] )
                && ( !global_dofs_seen[local_dof_indices[local_dof]] ) )
              {
                local_dofs_seen[local_dof] = true;
                global_dofs_seen[local_dof_indices[local_dof]] = true;
                all_renumberings[gradient_block0][local_dof_indices[local_dof]]
                 = all_dof_counts[gradient_block0];
                all_reordered_dofs[gradient_block0][local_dof_indices[local_dof]] = true;
                ++all_dof_counts[gradient_block0];
              }
              else if ( (fe.system_to_base_index(local_dof).first == block1_base_indices)
                && (fe.system_to_base_index(local_dof).second % fe.base_element(base_element).dofs_per_line > 0)
                && ( !local_dofs_seen[local_dof] )
                && ( !global_dofs_seen[local_dof_indices[local_dof]] ) )
              {
                local_dofs_seen[local_dof] = true;
                global_dofs_seen[local_dof_indices[local_dof]] = true;
                all_renumberings[gradient_block1][local_dof_indices[local_dof]]
                 = all_dof_counts[gradient_block1];
                all_reordered_dofs[gradient_block1][local_dof_indices[local_dof]] = true;
                ++all_dof_counts[gradient_block1];
              }
            }
            // Loop over all face DoFs, skipping the edge DoFs, stopping
            // before we reach the interior cell DoFs.
            // The ordering of these local dofs is by face, then gradients first, followed by non-gradients.
            for (unsigned int local_dof = line_dofs_per_cell;
                 local_dof < line_dofs_per_cell + face_dofs_per_cell; ++local_dof)
            {
              // First determine the block, then determine if we have a gradient or non-gradient.
              if ( (fe.system_to_base_index(local_dof).first == block0_base_indices)
                && ( !local_dofs_seen[local_dof] )
                && ( !global_dofs_seen[local_dof_indices[local_dof]] ) )
              {
                local_dofs_seen[local_dof] = true;
                global_dofs_seen[local_dof_indices[local_dof]] = true;
                // To decide if we have a gradient or non-gradient can use
                // the base element's local dof mod dofs_per_face. (after being adjusted to ignore lines).
                // Since gradients come first on each face:
                // If this is between 0 and n_gradients_per_face => gradient,
                //                                     otherwise => non-gradient.
                if ( (fe.system_to_base_index(local_dof).second - lines_per_cell*fe.base_element(base_element).dofs_per_line)
                     % fe.base_element(base_element).dofs_per_quad < n_gradients_per_face )
                {
                  all_renumberings[gradient_block0][local_dof_indices[local_dof]]
                   = all_dof_counts[gradient_block0];
                  all_reordered_dofs[gradient_block0][local_dof_indices[local_dof]] = true;
                  ++all_dof_counts[gradient_block0];
                }
                else
                {
                  all_renumberings[nongradient_block0][local_dof_indices[local_dof]]
                   = all_dof_counts[nongradient_block0];
                  all_reordered_dofs[nongradient_block0][local_dof_indices[local_dof]] = true;
                  ++all_dof_counts[nongradient_block0];
                }
              }
              else if ( (fe.system_to_base_index(local_dof).first == block1_base_indices)
                && ( !local_dofs_seen[local_dof] )
                && ( !global_dofs_seen[local_dof_indices[local_dof]] ) )
              {
                local_dofs_seen[local_dof] = true;
                global_dofs_seen[local_dof_indices[local_dof]] = true;
                // To decide if we have a gradient or non-gradient can use
                // the base element's local dof mod dofs_per_face. (after being adjusted to ignore lines).
                // Since gradients come first on each face:
                // If this is between 0 and n_gradients_per_face => gradient,
                //                                     otherwise => non-gradient.
                if ( (fe.system_to_base_index(local_dof).second - lines_per_cell*fe.base_element(base_element).dofs_per_line)
                     % fe.base_element(base_element).dofs_per_quad < n_gradients_per_face )
                {
                  all_renumberings[gradient_block1][local_dof_indices[local_dof]]
                   = all_dof_counts[gradient_block1];
                  all_reordered_dofs[gradient_block1][local_dof_indices[local_dof]] = true;
                  ++all_dof_counts[gradient_block1];
                }
                else
                {
                  all_renumberings[nongradient_block1][local_dof_indices[local_dof]]
                   = all_dof_counts[nongradient_block1];
                  all_reordered_dofs[nongradient_block1][local_dof_indices[local_dof]] = true;
                  ++all_dof_counts[nongradient_block1];
                }
              }
            }
            // Loop over all cell DoFs, skipping the edge and face DoFs,
            // stopping at the last local DoF
            // The ordering of these local dofs is simply by gradients followed by non-gradients.
            for (unsigned int local_dof = line_dofs_per_cell + face_dofs_per_cell;
                 local_dof < dofs_per_cell; ++local_dof)
            {
              if ( (fe.system_to_base_index(local_dof).first == block0_base_indices)
                && ( !local_dofs_seen[local_dof] )
                && ( !global_dofs_seen[local_dof_indices[local_dof]] ) )
              {
                local_dofs_seen[local_dof] = true;
                global_dofs_seen[local_dof_indices[local_dof]] = true;
                // To decide if we have a gradient or non-gradient, just check if the
                // base element's local dof is less than n_gradients_per_cell.
                // (after being shifted to only consider cell interiors).
                if ( (fe.system_to_base_index(local_dof).second
                      - lines_per_cell*fe.base_element(base_element).dofs_per_line
                      - faces_per_cell*fe.base_element(base_element).dofs_per_quad) < n_gradients_per_cell )
                {
                  all_renumberings[gradient_block0][local_dof_indices[local_dof]]
                   = all_dof_counts[gradient_block0];
                  all_reordered_dofs[gradient_block0][local_dof_indices[local_dof]] = true;
                  ++all_dof_counts[gradient_block0];
                }
                else
                {
                  all_renumberings[nongradient_block0][local_dof_indices[local_dof]]
                   = all_dof_counts[nongradient_block0];
                  all_reordered_dofs[nongradient_block0][local_dof_indices[local_dof]] = true;
                  ++all_dof_counts[nongradient_block0];
                }
              }
              else if ( (fe.system_to_base_index(local_dof).first == block1_base_indices)
                && ( !local_dofs_seen[local_dof] )
                && ( !global_dofs_seen[local_dof_indices[local_dof]] ) )
              {
                local_dofs_seen[local_dof] = true;
                global_dofs_seen[local_dof_indices[local_dof]] = true;
                // To decide if we have a gradient or non-gradient, just check if the
                // base element's local dof is less than n_gradients_per_cell.
                // (after being shifted to only consider cell interiors).
                if ( (fe.system_to_base_index(local_dof).second
                      - lines_per_cell*fe.base_element(base_element).dofs_per_line
                      - faces_per_cell*fe.base_element(base_element).dofs_per_quad) < n_gradients_per_cell )
                {
                  all_renumberings[gradient_block1][local_dof_indices[local_dof]]
                   = all_dof_counts[gradient_block1];
                  all_reordered_dofs[gradient_block1][local_dof_indices[local_dof]] = true;
                  ++all_dof_counts[gradient_block1];
                }
                else
                {
                  all_renumberings[nongradient_block1][local_dof_indices[local_dof]]
                   = all_dof_counts[nongradient_block1];
                  all_reordered_dofs[nongradient_block1][local_dof_indices[local_dof]] = true;
                  ++all_dof_counts[nongradient_block1];
                }
              }
            }
          }
        }
      }
      // Fill the array with counts of each type of DoF
      dof_counts.resize(n_blocks);
      for (unsigned int b=0; b<n_blocks; ++b)
      {
        dof_counts[b] = all_dof_counts[b];
      }

      // Now have reordering for lowest and higher order (split into gradients/non-gradients
      // with edge/face/cell subtypes) dofs.
      // Now want to combine them into a single reordering
      for (unsigned int dof = 0; dof < n_global_dofs; ++dof)
      {
        // Need to keep a track of the dof counts in blocks we've skipped
        // and adjust the renumbering accordingly.
        unsigned int block_increment = 0;
        for (unsigned int b=0; b<n_blocks; ++b)
        {
          if (all_reordered_dofs[b][dof])
          {
            renumbering[dof] = all_renumberings[b][dof] + block_increment;
            break;
          }
          block_increment += all_dof_counts[b];
        }
      }
      /* Check if we've now reordered all DoFs */
      unsigned int total_counted_dofs=0;
      for (unsigned int b=0; b<n_blocks; ++b)
      {
        total_counted_dofs += all_dof_counts[b];
      }

      if (total_counted_dofs == n_global_dofs)
      {
        return total_counted_dofs;
      }
      // If not, then add remaining DoFs in the order they come:
      std::cout << " DoFRenumbering::my_renumbering: some DoFs still remain..." << std::endl;
      for (unsigned int dof = 0; dof < n_global_dofs; ++dof)
      {
        if (!global_dofs_seen[dof])
        {
          renumbering[dof] = total_counted_dofs;
          ++total_counted_dofs;
          global_dofs_seen[dof] = true;
        }
      }
      return total_counted_dofs;
    }
  }
  // Short function to call the internal function:
  template <int dim, class DH>
  void my_renumbering (DH &dof_handler,
                     std::vector<unsigned int> &dof_counts)
  {
    std::vector<types::global_dof_index> renumbering(dof_handler.n_dofs());
    
    types::global_dof_index result = internals::
                                     compute_my_renumbering<dim, DH>(renumbering,
                                                                     dof_handler,
                                                                     dof_counts);
    if (result == 0)
    {
      return;
    }
    dof_handler.renumber_dofs (renumbering);
  }
  
  template
  void my_renumbering<3>(DoFHandler<3> &,
                      std::vector<unsigned int> &);
  
  
}