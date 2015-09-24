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

#ifndef MYPRECONDITIONER_H
#define MYPRECONDITIONER_H

// deal.II includes:
#include <deal.II/base/smartpointer.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/sparse_decomposition.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

// std includes:

// My includes:
#include <all_data.h>

using namespace dealii;

namespace Preconditioner
{
  // base class for all eddy current preconditioners:
  class EddyCurrentPreconditionerBase : public Subscriptor
  {
  public:
    EddyCurrentPreconditionerBase (const BlockSparseMatrix<double> &A);
    
    virtual void vmult (BlockVector<double> &dst, const BlockVector<double> &src) const;
    virtual void Tvmult (BlockVector<double> &dst, const BlockVector<double> &src) const;
  protected:
    const SmartPointer<const BlockSparseMatrix<double> > precon_matrix;
  };

  // Preconditioner for iterative solver, made up of a single block.
  // This essentially assumes the use of p=0.
  class EddyCurrentPreconditioner_1x1_lowOrder : public virtual EddyCurrentPreconditionerBase
  {
  public:
    EddyCurrentPreconditioner_1x1_lowOrder (const BlockSparseMatrix<double> &A);

    virtual void vmult (BlockVector<double> &dst, const BlockVector<double> &src) const;
    virtual void Tvmult (BlockVector<double> &dst, const BlockVector<double> &src) const;
  private:
    // Various options for the single block.
//     PreconditionIdentity block0;
//     PreconditionJacobi<> block0;
//     SparseILU<double> block0;
//     SparseILU<double>::AdditionalData block0_data;
//     PreconditionSOR<SparseMatrix<double> > block0;
    SparseDirectUMFPACK block0;
  };

  // Preconditioning for use with an FE_Nedelec-based 2x2 block matrix
  // made up of:
  // block 0: lowest
  // block 1: higher order edges/faces/cells.
  class EddyCurrentPreconditioner_2x2_lowHighOrder : public EddyCurrentPreconditionerBase
  {
  public:
    EddyCurrentPreconditioner_2x2_lowHighOrder (const BlockSparseMatrix<double> &A);

    void vmult (BlockVector<double> &dst, const BlockVector<double> &src) const;
    void Tvmult (BlockVector<double> &dst, const BlockVector<double> &src) const;
  private:
    // Block 0: lowest order edges.    
//     PreconditionIdentity block0;
//     PreconditionJacobi<> block0;
//     SparseILU<double> block0;
//     SparseILU<double>::AdditionalData block0_data;
//     PreconditionSOR<SparseMatrix<double> > block0;
    SparseDirectUMFPACK block0;

    // Block 1: higher order edges/faces/cells (or just edges if Block2/3 enabled:
//     PreconditionIdentity block1;
//      PreconditionJacobi<> block1;
    SparseILU<double> block1;
    SparseILU<double>::AdditionalData block1_data;
//     PreconditionSOR<SparseMatrix<double> > block1;
//    SparseDirectUMFPACK block1;
  };

  // Preconditioning for use with an FE_Nedelec-based 3x3 block matrix
  // made up of:
  // block 0: lowest
  // block 1: gradient-based higher order edges/faces/cells
  // block 2: non-gradient-based higher order edges/faces/cells
  // TODO: Make use of a nested CG solver for the higher order blocks.
  class EddyCurrentPreconditioner_3x3_lowHighOrderGradients : public EddyCurrentPreconditionerBase
  {
  public:
    EddyCurrentPreconditioner_3x3_lowHighOrderGradients (const BlockSparseMatrix<double> &A);

    void vmult (BlockVector<double> &dst, const BlockVector<double> &src) const;
    void Tvmult (BlockVector<double> &dst, const BlockVector<double> &src) const;
  private:
    // Block 0: lowest order:
//     PreconditionIdentity block0;
//     PreconditionJacobi<> block0;
//     SparseILU<double> block0;
//     SparseILU<double>::AdditionalData block0_data;
//     PreconditionSOR<SparseMatrix<double> > block0;
    SparseDirectUMFPACK block0;

    // Block 1: higher order gradients:
//     PreconditionIdentity block1;
//     PreconditionJacobi<> block1;
    SparseDirectUMFPACK block1;
//     PreconditionSOR<SparseMatrix<double> > block1;
//     SparseILU<double> block1;
//     SparseILU<double>::AdditionalData block1_data;

    // Block 2: higher order non-gradients:
//     PreconditionIdentity block2;
//     PreconditionJacobi<> block2;
    SparseDirectUMFPACK block2;
//     PreconditionSOR<SparseMatrix<double> > block2;
//     SparseILU<double> block2;
//     SparseILU<double>::AdditionalData block2_data;
  };
  
  // Preconditioning for use with an FE_Nedelec-based 3x3 block matrix
  // made up of:
  // block 0: lowest
  // block 1: gradient-based higher order edges/faces/cells
  // block 2: non-gradient-based higher order edges/faces/cells
  // TODO: Make use of a nested CG solver for the higher order blocks.
  class EddyCurrentPreconditioner_5x5 : public EddyCurrentPreconditionerBase
  {
  public:
    EddyCurrentPreconditioner_5x5 (const BlockSparseMatrix<double> &A);

    void vmult (BlockVector<double> &dst, const BlockVector<double> &src) const;
    void Tvmult (BlockVector<double> &dst, const BlockVector<double> &src) const;
  private:
    // Block 0: lowest order:
//     PreconditionIdentity block0;
//     PreconditionJacobi<> block0;
//     SparseILU<double> block0;
//     SparseILU<double>::AdditionalData block0_data;
//     PreconditionSOR<SparseMatrix<double> > block0;
    SparseDirectUMFPACK block0;

    // Block 1: real gradients:
//     PreconditionIdentity block1;
//     PreconditionJacobi<> block1;
    SparseDirectUMFPACK block1;
//     PreconditionSOR<SparseMatrix<double> > block1;
//     SparseILU<double> block1;
//     SparseILU<double>::AdditionalData block1_data;

    // Block 2: real non-gradients:
//     PreconditionIdentity block2;
//     PreconditionJacobi<> block2;
    SparseDirectUMFPACK block2;
//     PreconditionSOR<SparseMatrix<double> > block2;
//     SparseILU<double> block2;
//     SparseILU<double>::AdditionalData block2_data;
    
    // NOTE: Don't actually need block3 or block4
    // They will match block1 and block2, so can just reuse them.
    // Block 3: imag gradients:
//     PreconditionIdentity block3;
//     PreconditionJacobi<> block3;
//     SparseDirectUMFPACK block3;
//     PreconditionSOR<SparseMatrix<double> > block3;
//     SparseILU<double> block3;
//     SparseILU<double>::AdditionalData block3_data;
    
    // Block 4: imag non-gradients:
//     PreconditionIdentity block4;
//     PreconditionJacobi<> block4;
//     SparseDirectUMFPACK block4;
//     PreconditionSOR<SparseMatrix<double> > block4;
//     SparseILU<double> block4;
//     SparseILU<double>::AdditionalData block4_data;
  };
}
#endif