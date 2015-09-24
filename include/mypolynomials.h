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

#ifndef MYPOLYNOMIALS_H
#define MYPOLYNOMIALS_H

#include <deal.II/base/polynomial.h>
#include <deal.II/base/point.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/thread_management.h>

#include <cmath>
#include <algorithm>
#include <limits>

namespace MyPolynomials
{
  using namespace dealii;
  using namespace Polynomials;

  // Copy of the original deal.II legendre class, but with the coefficents adjusted
  // so that the recursive formula is for the integrated legendre polynomials described
  // in the thesis of Sabine Zaglmayer.
  //
  // L0 = -1  (added so that it can be generated recursively from 0)
  // L1 = x
  // L2 = 0.5*(x^2 - 1)
  // (n+1)Lnp1 = (2n-1)xLn - (n-2)Lnm1.
  //
  // However, it is possible to generate them directly from the legendre polynomials:
  //
  // (2n-1)Ln = ln - lnm2
  //
  // Could simply call the legendre set and adjust accordingly??
  class integratedLegendre : public Polynomial<double>
  {
  public:
    integratedLegendre (unsigned int p);
    
    static
    std::vector<Polynomial<double> >
    generate_complete_basis (const unsigned int degree); 
    
  private:
    static std::vector<std_cxx11::shared_ptr<const std::vector<double> > > shifted_coefficients;
    
    static std::vector<std_cxx11::shared_ptr<const std::vector<double> > > recursive_coefficients;

    static void compute_coefficients (const unsigned int p);
    
    static const std::vector<double> &
    get_coefficients (const unsigned int k);
    
  };
}

#endif