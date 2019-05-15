/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(ovrvo,FixOVRVO)

#else

#ifndef LMP_FIX_OVRVO_H
#define LMP_FIX_OVRVO_H

#include "fix.h"

namespace LAMMPS_NS {

class FixOVRVO : public Fix {
 public:
  FixOVRVO(class LAMMPS *, int, char **);
  virtual ~FixOVRVO() {}
  int setmask();
  virtual void init();
  virtual void initial_integrate(int);
  virtual void final_integrate();

 protected:
  class RanPark *random;
  double dt;
  double t_target;
  double gamma;
  double o_coeff_v, o_coeff_n, v_coeff_f, r_coeff_v;
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
