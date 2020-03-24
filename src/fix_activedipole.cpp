/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.

   This file is part of custom LAMMPS code for simulations of active matter.
    author: Cory Hargus
    e-mail: hargus@berkeley.edu
    github: https://github.com/mandadapu-group/active-matter
------------------------------------------------------------------------- */
#include "stdio.h"
#include "string.h"
#include "fix_activedipole.h"
#include "atom.h"
#include "neighbor.h"
#include "force.h"
#include "update.h"
#include "error.h"
#include "math.h"

using namespace LAMMPS_NS;
using namespace FixConst;

enum{CCW,CW,MIXED,CONVECT};

/* ---------------------------------------------------------------------- */

// example command:
// fix activedipole TORQUE_MAGNITUDE TORQUE_TYPE
FixActiveDipole::FixActiveDipole(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (strcmp(style,"activedipole") != 0 && narg < 4)
    error->all(FLERR,"Illegal fix activedipole command: not enough args");
  f_active = force->numeric(FLERR,arg[3]);
  activestyle = CCW;
  if (narg == 5){
    if (strcmp(arg[4],"ccw") == 0)
      activestyle = CCW;
    else if (strcmp(arg[4],"cw") == 0)
      activestyle = CW;
    else if (strcmp(arg[4],"mixed") == 0)
      activestyle = MIXED;
    else if (strcmp(arg[4],"convect") == 0)
      activestyle = CONVECT;
    else
      error->all(FLERR, "Only {ccw, cw, mixed, convect} are accepted styles.");
  }
}

/* ---------------------------------------------------------------------- */

int FixActiveDipole::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixActiveDipole::post_force(int /*vflag*/)
{
  int nlocal = atom->nlocal;
  double **x = atom->x;
  double **f = atom->f;
  double **torque = atom->torque;


  // Add the active force to the already-computed per-atom forces
  for (int i = 0; i < nlocal; i++) {
    if (activestyle==CCW){
      torque[i][2] += f_active;
    }

    else if (activestyle==CW){
      torque[i][2] += f_active;
    }

    else if (activestyle==MIXED){
      if (i % 2 == 0){
        torque[i][2] += f_active;
      }
      else{
        torque[i][2] += f_active;
      }
    // else if (activestyle==CONVECT){
    //   // Apply convective force in direction of dipole
    // }
    }
  }
}
