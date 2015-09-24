An MIT Forward Solver
=====================

This code is provided to solve the eddy current approximation of the time-harmonic Maxwell equations.

It makes use of H(curl)-conforming hp-finite element methods at arbitrary polynomial order.

The deal.II library, found at http://www.dealii.org is required for this code to run.

Assuming deal.II 8.3 is installed and configured properly, then the code should run successfully. However, we recommend that the deal.II development branch dated July 6th (SHA hash 79583e56) is used to ensure total compatibility.

Usage:
--------

As an example of usage, we refer to the case of a conducting sphere in a uniform background field, as detailed in [1].

To run the sphere benchmark code:

    $ cd benchmark/sphere_benchmark
    $ mkdir build
    $ cd build
    $ cmake ../
    $ make
    $ ./sphere_benchmark -i ../input_files/sphere_team6_dealmesh.prm -m 2 -h 1 -p 1

This will run the TEAM 6 benchmark on a coarse mesh (1 global refinement of the primitive deal mesh with a second-order mapping) at polynomial order 1 and produce a .vtk file which can be visualised using software such as PARAVIEW as well as further data files which can be investigated using software such as MATLAB or python scripts.

There are further command line options which can be seen in the source of the benchmark code.

It is wise to consider the memory requirements of the code when considering fine meshes and higher polynomial orders.

Work derived from this software:
--------
In addition to the terms of the GPL v3 license (see below), we kindly ask that any work using or derived from this code cites the following papers:

[1] R.M. Kynch, P.D. Ledger. Resolving the sign conflict problem for hp hexahedral edge elements with applications to eddy current problems, submitted 2015.

[2] R.M. Kynch, P.D. Ledger. TBC

[3] P.D. Ledger, S. Zaglmayr, hp-Finite element simulation of three-diemensional eddy current problems on multiply connected domains. Computer Methods in Applied Mechanics and Engineering 199 (2010) 3386-3401.

[4] J. Schoeberl and S. Zaglmayr,  High order Nedelec elements with local complete exact sequence properties Int. J. Comput. Math. Electr. Electron. Engrg (COMPEL) 24 (2005) 374-384.

and, if appropriate, co-authorship of any subsequent publications be offered to the authors of this software.

License:
--------

The code in this repository is provided under the GPL v3 license, please see the file ./LICENSE for details
