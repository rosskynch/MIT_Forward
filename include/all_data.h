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

#ifndef ALL_DATA_H
#define ALL_DATA_H

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>


// Collection of data namespaces for use in the various other codes.
// It may be wise to set these up as structures for each class (TODO),
// but for now they will suffice.

using namespace dealii;

namespace EquationData
{
  // Namespace containing data for use with the EddyCurrent class
  // TODO: clean up and possibly rename to "EddyCurrentData" to make it clear where it is used??
  
  // Data for use with equations of the form:
  // curl((1/mu_r)*curl(A)) + kappa*A = rhs_factor*J_s.
  
  // Electromagnetic constants:
  extern const double constant_epsilon0; // electric constant (permittivity)
  extern const double constant_mu0;//1.25663706e-6; // magnetic constant (permeability)
  extern const double constant_sigma0; // background conductivity, set to regularise system (can't solve curlcurlE = f).
  
  // Material Parameters:
  extern double param_omega; // angular frequency, rads/sec.
  extern double param_regularisation;
  
  // Material parameters for non-conducting region:
  extern double param_epsilon_background;
  extern double param_sigma_background;
  extern double param_mur_background;
  
  // 
  // Material parameter of the conducting object
  // TODO: Eventually this will be removed 
  //       when we're interested in more parameters.
  //
  extern double param_epsilon_conducting;
  extern double param_sigma_conducting;
  extern double param_mur_conducting;  
  
  /* Vectors holding equation parameters
   * intially assume 1 objects.
   * This can be adjusted via the parameter file and then
   * use vector.reinit(number_of objects) later on.
   */
  extern Vector<double> param_mur;
  extern Vector<double> param_sigma;
  extern Vector<double> param_epsilon;
  // Kappa = Kappa_re + i*Kappa_im = -omega.^2*epr + i*omega*sigma
  extern Vector<double> param_kappa_re; 
  extern Vector<double> param_kappa_im;
  
  // Factor for the RHS of the equation.
  extern double rhs_factor;
  
  extern bool neumann_flag; // true = neumann, false = dirichlet.
}

namespace PolarizationTensor
{
  extern bool enable;
  extern bool takeComplexConj;
  extern std::vector< FullMatrix<double> > polarizationTensor;
  extern std::vector< Vector<double> > H0;
}

namespace TEAMBenchmark
{
  // Data for the team benchmark problem.
  extern bool enable;
  extern Point<3> coil_centre;
  extern std::vector<Point<3>> corner_centres;
  extern types::material_id coil_material_id;
}

namespace MeshData
{
  extern bool external_mesh;
  
  extern std::string mesh_filename;
  
  extern std::string boundary_shape;
  
  // For cubes/cuboids/boxes/etc:
  extern double xmax;
  extern double ymax;
  extern double zmax;
  extern double xmin;
  extern double ymin;
  extern double zmin;
  
  // For cylinders/spheres
  extern double height;
  extern double radius;
  extern double inner_radius;
  
  // For Toruses
  extern double major_radius;
  extern double minor_radius;
  
  extern Point<3> centre;
  
}


namespace IO_Data
{
  // Input/output data for use with the input/output classes
  // TODO: look at splitting into two namespaces (1 input and 1 output).
  // IS THIS EVEN REQUIRED ANYMORE ???
  extern std::string mesh_filename;
  extern std::string parameter_filename;
  extern std::string output_filename;
  extern std::string output_filetype;
  extern unsigned int n_subdivisions;
}

namespace PreconditionerData
{
  extern bool use_direct;
  extern double solver_tolerance;
  extern double strengthen_diagonal;
  extern unsigned int extra_off_diagonals;
  extern bool right_preconditioning;
  
  extern bool constrain_gradients;
}

namespace ExcitationCoilData
{
  // Contains all information about the excitation coils.
  // This must be populated later in the code, either
  // directly or via some input.
  // This data will be used by the InverseSolver class, so must be
  // filled first before calling routines in that class.
  // TODO: alternative is to pass a structure with the data to that routine??
    
  // This data will be filled in by the input parameters
  // it can then be used to fill the coil position/direction vectors.
  extern unsigned int number_of_coils;
  extern Tensor<1,3> array_centre;
  extern double array_radius;
  extern double array_angle;
  extern double coil_radius;
    
  extern std::vector<Point<3>> coil_position;
  extern std::vector<Tensor<1,3>> coil_direction;
  
}

namespace SensorCoilData
{
  // Contains all information about sensor coils.
  // This must be populated later in the code, either
  // directly or via some input.
  // This data will be used by the InverseSolver class, so must be
  // filled first before calling routines in that class.
  // TODO: alternative is to pass a structure with the data to that routine??
    
  // This data will be filled in by the input parameters
  // it can then be used to fill the coil position/direction vectors.
  extern unsigned int number_of_coils;
  extern Tensor<1,3> array_centre;
  extern double array_radius;
  extern double array_angle;
  extern double coil_radius;
  
  extern std::vector<Point<3>> coil_position;
  extern std::vector<Tensor<1,3>> coil_direction;
}

namespace InverseProblemData
{
  // Data specific to the inverse method
  extern bool use_all; // not in the input file for now - update later if we need it?? (TODO)
  extern unsigned int n_voxels; // Will be calculated when processing the mesh
  extern unsigned int n_coil_combinations; // calculated from input data (number of sensor coils x number of excitation coils).
 
  // material_id within the input mesh for use in the setup routine.
  extern types::material_id recovery_region_id;
  extern types::material_id background_region_id;
  
  extern std::vector<Vector<double>> measured_voltages_re;
  extern std::vector<Vector<double>> measured_voltages_im;
  
  // G-N data:
  extern double gn_step_size;
  extern double gn_regularisation_parameter;
  extern unsigned int gn_max_iterations;
  // Storage for Gauss Newton history:
  extern Vector<double> initial_sigma;
  
  // TESTING: keep count of GN iterations.  
  extern unsigned int iteration_count;
}
#endif