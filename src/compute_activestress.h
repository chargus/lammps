/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(activestress,ComputeActiveStress)

#else

#ifndef LMP_COMPUTE_ACTIVESTRESS_H
#define LMP_COMPUTE_ACTIVESTRESS_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeActiveStress : public Compute {
 public:
  ComputeActiveStress(class LAMMPS *, int, char **);
  virtual ~ComputeActiveStress();
  void init() {}
  // void setup();
  // virtual double compute_scalar();
  virtual void compute_vector();
  // virtual void compute_array();
 protected:
  double nktv2p,inv_volume;
  virtual void kinetic_compute();
  virtual void virial_compute();
  virtual void active_compute();

};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

*/