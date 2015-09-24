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

#ifndef MYFE_NEDELEC_H
#define MYFE_NEDELEC_H

#include <deal.II/base/derivative_form.h>
#include <deal.II/base/qprojector.h>
#include <deal.II/base/polynomials_bdm.h>
#include <deal.II/base/polynomials_raviart_thomas.h>
#include <deal.II/base/polynomials_abf.h>
#include <deal.II/base/polynomials_nedelec.h>
#include <deal.II/fe/fe_poly_tensor.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_cartesian.h>

#include <mypolynomials.h>

// TODO: remove use of polyspace, it causes too many issues when combining sigma and lambda
//       instead, use the 1D polynomial and generate the required derivatives manually - it's very simple anyway.
//

using namespace dealii;
template <int dim>
class MyFE_Nedelec : public FiniteElement<dim,dim>
// Note: spacedim=dim.. have removed all templates using spacedim
//       for this element. To undo, would have to compare with FE_PolyTensor
//       or look at the git commit.
{
public:
  // Constructor
  MyFE_Nedelec (const unsigned int degree);
//                 const FiniteElementData<dim> &fe_data,
//                 const std::vector<bool> &restriction_is_additive_flags,
//                 const std::vector<ComponentMask> &nonzero_components);
  
  virtual std::string get_name () const;
  
  virtual FiniteElement<dim> *clone() const;
  
  // This element is vector-valued so throw an exception:
  virtual double shape_value (const unsigned int i,
                              const Point<dim> &p) const;

  virtual double shape_value_component (const unsigned int i,
                                        const Point<dim> &p,
                                        const unsigned int component) const;

  // This element is vector-valued so throw an exception:
  virtual Tensor<1,dim> shape_grad (const unsigned int  i,
                                    const Point<dim>   &p) const;

  virtual Tensor<1,dim> shape_grad_component (const unsigned int i,
                                              const Point<dim> &p,
                                              const unsigned int component) const;

  // This element is vector-valued so throw an exception:
  virtual Tensor<2,dim> shape_grad_grad (const unsigned int  i,
                                         const Point<dim> &p) const;

  virtual Tensor<2,dim> shape_grad_grad_component (const unsigned int i,
                                                   const Point<dim> &p,
                                                   const unsigned int component) const;

  /**
   * Given <tt>flags</tt>, determines the values which must be computed only
   * for the reference cell. Make sure, that #mapping_type is set by the
   * derived class, such that this function can operate correctly.
   */
  virtual UpdateFlags update_once (const UpdateFlags flags) const;
  /**
   * Given <tt>flags</tt>, determines the values which must be computed in
   * each cell cell. Make sure, that #mapping_type is set by the derived
   * class, such that this function can operate correctly.
   */
  virtual UpdateFlags update_each (const UpdateFlags flags) const;
  
  

  
 
protected:
  /**
   * The mapping type to be used to map shape functions from the reference
   * cell to the mesh cell.
   */
  MappingType mapping_type;

  virtual
  typename Mapping<dim,dim>::InternalDataBase *
  get_data (const UpdateFlags,
            const Mapping<dim,dim> &mapping,
            const Quadrature<dim> &quadrature) const ;

  virtual void
  fill_fe_values (const Mapping<dim,dim>                               &mapping,
                  const typename Triangulation<dim,dim>::cell_iterator &cell,
                  const Quadrature<dim>                                &quadrature,
                  typename Mapping<dim,dim>::InternalDataBase          &mapping_data,
                  typename Mapping<dim,dim>::InternalDataBase          &fedata,
                  FEValuesData<dim,dim>                                &data,
                  CellSimilarity::Similarity                           &cell_similarity) const;

  virtual void
  fill_fe_face_values (const Mapping<dim,dim>                               &mapping,
                       const typename Triangulation<dim,dim>::cell_iterator &cell,
                       const unsigned int                                    face_no,
                       const Quadrature<dim-1>                              &quadrature,
                       typename Mapping<dim,dim>::InternalDataBase          &mapping_data,
                       typename Mapping<dim,dim>::InternalDataBase          &fedata,
                       FEValuesData<dim,dim>                                &data) const;

  virtual void
  fill_fe_subface_values (const Mapping<dim,dim>                               &mapping,
                          const typename Triangulation<dim,dim>::cell_iterator &cell,
                          const unsigned int                                    face_no,
                          const unsigned int                                    sub_no,
                          const Quadrature<dim-1>                              &quadrature,
                          typename Mapping<dim,dim>::InternalDataBase          &mapping_data,
                          typename Mapping<dim,dim>::InternalDataBase          &fedata,
                          FEValuesData<dim,dim> &data) const;
                          
                          
  // DERVIED INTERNAL DATA - CAN BE USED TO STORE PRECOMPUTED INFO
  // E.G. Once dim is know we can compute sigma, lambda, and combinations.
  class InternalData : public FiniteElement<dim>::InternalDataBase
  {
  public:
    // Storage for shape functions on the reference element
    // We only pre-compute those cell-based DoFs, as the edge-based
    // dofs depend on the choice of cell.
    std::vector<std::vector<Tensor<1,dim> > > shape_values;
    
    std::vector< std::vector< DerivativeForm<1, dim, dim> > > shape_grads;
    
    // TODO: remove the redundant parts, will need to update the 2D case when this is removed.
    // - sigma - these won't be reused.
    // - lambda - won't be reused.
    // - lambda_ipj = lambda[i] + lambda[j] - no need, can go straight to the edge_lambda_*.
    // - edge_sigma_values/grads - these can be formed straight from sigma_imj_* - no need to store them in data, they can just be created in fill_edge_values.
    // - edgeDoF_to_poly - should remove the use of polyspace and just use the 1d polynomial for everything.
    
    
    // They should run like so: sigma/lambda[q][i] returns the sigma associated with vertex i at the quad point q.
    std::vector<std::vector<double> > sigma; // remove.
    std::vector<std::vector<double> > lambda; // remove.
    // lambda_ipj and sigma_imj represent combinations of addtion and subtraction
    // e.g. lambda_iph[q][i][j] correspond to lambda[i]+lambda[j] at quad point q.
    std::vector<std::vector<std::vector<double> > > lambda_ipj; // remove
    
    std::vector<std::vector<std::vector<double> > > sigma_imj_values;
    std::vector<std::vector<std::vector<double> > > sigma_imj_grads;
    std::vector<std::vector<int> > sigma_imj_sign;
    std::vector<std::vector<unsigned int> > sigma_imj_component;
    
    // TODO: join the 3D/2D versions.
    
    // better way is to store the "standard" edge sigma & lambda
    // then in the cell it can be adjusted.
    // On edge, E_m = [e_{1}^{m}, e_{2}^{m}].
    // edge_sigma[m][q] = sigma[e2][q] - sigma[e1][q]
    // edge_lambda_values[m][q] = lambda[e1][q] + lambda[e1][q]
    //
    // Note that the gradient components of edge_sigma are constant,
    // so we don't need the values at every quad point, whereas they are
    // needed for the lambda gradient.
    std::vector<std::vector<double> > edge_sigma_values; // remove & handle only in fill_edge_values.
    std::vector<std::vector<double> > edge_lambda_values;
    
    std::vector<std::vector<double> > edge_sigma_grads; // remove & handle only in fill_edge_values.
    // need additional dimension in 3D for the lambda grads.
    std::vector<std::vector<double> > edge_lambda_grads_2d;
    std::vector<std::vector<std::vector<double> > > edge_lambda_grads_3d;
    
    // There are non-zero second derivatives for lambda,
    // but they are constant across the cell.
    std::vector<std::vector<std::vector<double> > > edge_lambda_gradgrads_3d;
    
    // TODO: add face parameterisation storage:
    // eta, xi
    std::vector<std::vector<double> > face_lambda_values;
    std::vector<std::vector<double> > face_lambda_grads;
    
    
    // TODO: remove this use of this from 2D:
    // These will be used to map through the polynomial space
    // In order to get to the basis functions we require.
    // edgeDoF_to_poly[m][i] will return the x/y/z-component of polynomial space
    // to look for degree of freedom i on edge m.
    // each edge will have DoFs, i=[0, degree].
    // Note that the zero-th order polynomials are generated outside of the polynomial space.
    
    std::vector<std::vector<unsigned int> > edgeDoF_to_poly; // remove - will need to switch over 2D edge-based DoFs (gradients) to use 1d polynomials.
  };
  
private:
  
  static std::vector<unsigned int> get_dpo_vector (unsigned int degree);
  
  std::vector<Polynomials::Polynomial<double> > polynomials_1d;
  
  std::vector<std::vector< Polynomials::Polynomial<double> > >
    create_polynomials (const unsigned int degree);
  
  const AnisotropicPolynomials<dim> polynomial_space; // remove
  
  // returns the number of dofs per vertex/edge/face/cell
  std::vector<unsigned int> get_n_pols(unsigned int degree);
  
  // returns the number of polynomials in the basis set.
  unsigned int compute_n_pols (unsigned int degree);
  
  // Calculates the edge_based shape functions
  // which depend on the cell.
  void fill_edge_values(const typename Triangulation<dim,dim>::cell_iterator &cell,  
                        const Quadrature<dim>                                &quadrature,
                        InternalData                                         &fedata) const;  
                        
  void fill_face_values(const typename Triangulation<dim,dim>::cell_iterator &cell,  
                        const Quadrature<dim>                                &quadrature,
                        InternalData                                         &fedata) const;  
};


#endif
  