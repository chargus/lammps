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

FixStyle(dumbbell,FixDumbbell)

#else

#ifndef LMP_FIX_DUMBBELL_H
#define LMP_FIX_DUMBBELL_H

#include "fix.h"

namespace LAMMPS_NS {

class FixDumbbell : public Fix {
 public:
  FixDumbbell(class LAMMPS *, int, char **);
  virtual ~FixDumbbell() {}
  int setmask();
  virtual void init();
  virtual void final_integrate();

 protected:
  class RanPark *random;
  double dt,dtfm;
  double t_target;
  double gamma, gamma_i;
  double f_active;
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
