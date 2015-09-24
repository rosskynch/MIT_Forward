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

#include <mypreconditioner.h>

using namespace dealii;

namespace Preconditioner
{
  // base class:
  EddyCurrentPreconditionerBase::EddyCurrentPreconditionerBase (const BlockSparseMatrix<double> &A)
  :
  precon_matrix (&A)
  {
  }
  void EddyCurrentPreconditionerBase::vmult (BlockVector<double>       &dst,
                                             const BlockVector<double> &src) const
  {
    precon_matrix->vmult(dst,src);
  }
  void EddyCurrentPreconditionerBase::Tvmult (BlockVector<double>       &dst,
                                             const BlockVector<double> &src) const
  {
    precon_matrix->Tvmult(dst,src);
  }
  // END base class
  // EddyCurrentPreconditioner_1x1_lowOrder
  EddyCurrentPreconditioner_1x1_lowOrder::EddyCurrentPreconditioner_1x1_lowOrder (const BlockSparseMatrix<double> &A)
  :
  EddyCurrentPreconditionerBase(A)
  {
    // Block 0:
    //direct:
    block0.initialize(precon_matrix->block(0,0));

    // ILU:
//     block0_data.extra_off_diagonals = PreconditionerData::extra_off_diagonals;
//     block0_data.strengthen_diagonal = PreconditionerData::strengthen_diagonal;
//     block0.initialize(precon_matrix->block(0,0),
//                       block0_data);
  }
  void EddyCurrentPreconditioner_1x1_lowOrder::vmult (BlockVector<double>       &dst,
                                                      const BlockVector<double> &src) const
  {
    block0.vmult (dst.block(0), src.block(0));
  }
  void EddyCurrentPreconditioner_1x1_lowOrder::Tvmult (BlockVector<double>       &dst,
                                                       const BlockVector<double> &src) const
  {
    block0.Tvmult (dst.block(0), src.block(0));
  }
  // END EddyCurrentPreconditioner_1x1_lowOrder

  // EddyCurrentPreconditioner_2x2_lowAndHighOrder
  EddyCurrentPreconditioner_2x2_lowHighOrder::EddyCurrentPreconditioner_2x2_lowHighOrder (const BlockSparseMatrix<double> &A)
  :
  EddyCurrentPreconditionerBase(A)
  {
    // Block 0:
    //direct:
    block0.initialize(precon_matrix->block(0,0));

    // ILU:
//     block0_data.extra_off_diagonals = PreconditionerData::extra_off_diagonals;
//     block0_data.strengthen_diagonal = PreconditionerData::strengthen_diagonal;
//     block0.initialize(precon_matrix->block(0,0),
//                       block0_data);

    // SOR:
//     block0.initialize(precon_matrix->block(0,0),1e6);

    //Block 1:
    // Options for ILU:
    block1_data.extra_off_diagonals = PreconditionerData::extra_off_diagonals;
    block1_data.strengthen_diagonal = PreconditionerData::strengthen_diagonal;

    block1.initialize(precon_matrix->block(1,1),
                      block1_data);
  }
  void EddyCurrentPreconditioner_2x2_lowHighOrder::vmult (BlockVector<double>       &dst,
                                                          const BlockVector<double> &src) const
  {
    block0.vmult (dst.block(0), src.block(0));
    block1.vmult (dst.block(1), src.block(1));
  }
  void EddyCurrentPreconditioner_2x2_lowHighOrder::Tvmult (BlockVector<double>       &dst,
                                                           const BlockVector<double> &src) const
  {
    block0.Tvmult (dst.block(0), src.block(0));
    block1.Tvmult (dst.block(1), src.block(1));
  }
  // END EddyCurrentPreconditioner_2x2_lowHighOrder
  
  // EddyCurrentPreconditioner_3x3_lowHighOrderGradients
    EddyCurrentPreconditioner_3x3_lowHighOrderGradients::EddyCurrentPreconditioner_3x3_lowHighOrderGradients (const BlockSparseMatrix<double> &A)
  :
  EddyCurrentPreconditionerBase(A)
  {
    // Block 0:
    //direct:
    block0.initialize(precon_matrix->block(0,0));
    
    // ILU:
//     block0_data.extra_off_diagonals = PreconditionerData::extra_off_diagonals;
//     block0_data.strengthen_diagonal = PreconditionerData::strengthen_diagonal;
//     block0.initialize(precon_matrix->block(0,0),
//                       block0_data);

    //Block 1:
    // Options for ILU:
//     block1_data.extra_off_diagonals = PreconditionerData::extra_off_diagonals;
//     block1_data.strengthen_diagonal = PreconditionerData::strengthen_diagonal;

//     block1.initialize(precon_matrix->block(1,1),
//                       block1_data);
    block1.initialize(precon_matrix->block(1,1));
    
    //Block 2:
    // Options for ILU:
//     block2_data.extra_off_diagonals = PreconditionerData::extra_off_diagonals;
//     block2_data.strengthen_diagonal = PreconditionerData::strengthen_diagonal;

//     block2.initialize(precon_matrix->block(2,2),
//                       block2_data);
    block2.initialize(precon_matrix->block(2,2));

  }
  void EddyCurrentPreconditioner_3x3_lowHighOrderGradients::vmult (BlockVector<double>       &dst,
                                                                   const BlockVector<double> &src) const
  {
    block0.vmult (dst.block(0), src.block(0));
    block1.vmult (dst.block(1), src.block(1));
    block2.vmult (dst.block(2), src.block(2));
  }
  void EddyCurrentPreconditioner_3x3_lowHighOrderGradients::Tvmult (BlockVector<double>       &dst,
                                                                    const BlockVector<double> &src) const
  {
    block0.Tvmult (dst.block(0), src.block(0));
    block1.Tvmult (dst.block(1), src.block(1));
    block2.Tvmult (dst.block(2), src.block(2));
  }
  // END EddyCurrentPreconditioner_3x3_lowHighOrderGradients
  
// EddyCurrentPreconditioner_5x5
    EddyCurrentPreconditioner_5x5::EddyCurrentPreconditioner_5x5 (const BlockSparseMatrix<double> &A)
  :
  EddyCurrentPreconditionerBase(A)
  {
    // Block 0:
    //direct:
    block0.initialize(precon_matrix->block(0,0));
    
    // ILU:
//     block0_data.extra_off_diagonals = PreconditionerData::extra_off_diagonals;
//     block0_data.strengthen_diagonal = PreconditionerData::strengthen_diagonal;
//     block0.initialize(precon_matrix->block(0,0),
//                       block0_data);

    //Block 1:
    // Options for ILU:
//     block1_data.extra_off_diagonals = PreconditionerData::extra_off_diagonals;
//     block1_data.strengthen_diagonal = PreconditionerData::strengthen_diagonal;

//     block1.initialize(precon_matrix->block(1,1),
//                       block1_data);
    block1.initialize(precon_matrix->block(1,1));
    
    //Block 2:
    // Options for ILU:
//     block2_data.extra_off_diagonals = PreconditionerData::extra_off_diagonals;
//     block2_data.strengthen_diagonal = PreconditionerData::strengthen_diagonal;

//     block2.initialize(precon_matrix->block(2,2),
//                       block2_data);
    block2.initialize(precon_matrix->block(2,2));
    
    //Block 3:
    // Options for ILU:
//     block3_data.extra_off_diagonals = PreconditionerData::extra_off_diagonals;
//     block3_data.strengthen_diagonal = PreconditionerData::strengthen_diagonal;

//     block3.initialize(precon_matrix->block(3,3),
//                       block3_data);
//     block3.initialize(precon_matrix->block(3,3));
    
    //Block 4:
    // Options for ILU:
//     block4_data.extra_off_diagonals = PreconditionerData::extra_off_diagonals;
//     block4_data.strengthen_diagonal = PreconditionerData::strengthen_diagonal;

//     block4.initialize(precon_matrix->block(4,4),
//                       block4_data);
//     block4.initialize(precon_matrix->block(4,4));

  }
  void EddyCurrentPreconditioner_5x5::vmult (BlockVector<double>       &dst,
                                             const BlockVector<double> &src) const
  {
    block0.vmult (dst.block(0), src.block(0));
    block1.vmult (dst.block(1), src.block(3)); //block 3
    dst.block(1) *= -1.0;
    block2.vmult (dst.block(2), src.block(2));
    block1.vmult (dst.block(3), src.block(1)); //block 1
    block2.vmult (dst.block(4), src.block(4));
  }
  void EddyCurrentPreconditioner_5x5::Tvmult (BlockVector<double>       &dst,
                                              const BlockVector<double> &src) const
  {
    block0.Tvmult (dst.block(0), src.block(0));
    block1.Tvmult (dst.block(1), src.block(1)); //block 1
    block2.Tvmult (dst.block(2), src.block(2));
    block1.Tvmult (dst.block(3), src.block(3)); //block 3
    dst.block(3) *= -1.0;
    block2.Tvmult (dst.block(4), src.block(4));
  }
  // END EddyCurrentPreconditioner_5x5
}
// END namespace Preconditioner