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

#include "stdio.h"
#include "string.h"
#include "fix_dumbbell.h"
#include "atom.h"
#include "neighbor.h"
#include "force.h"
#include "update.h"
#include "error.h"
#include "math.h"
#include "random_park.h"

using namespace LAMMPS_NS;
using namespace FixConst;

enum{CCW,CW,MIXED,CONVECT};

/* ---------------------------------------------------------------------- */

// example command:
// fix dumbbell ACTIVEFORCE
FixDumbbell::FixDumbbell(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (strcmp(style,"dumbbell") != 0 && narg < 4)
    error->all(FLERR,"Illegal fix dumbbell command: not enough args");
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

int FixDumbbell::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixDumbbell::post_force(int /*vflag*/)
{
  double delx, dely, rsq, r;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  double **x = atom->x;
  double **f = atom->f;

  // Add the active force to the already-computed per-atom forces
  for (int n = 0; n < nbondlist; n++) {
    int i1 = bondlist[n][0];
    int i2 = bondlist[n][1];
    // type = bondlist[n][2];
    // Get a unit vector pointing from atom 1 to atom 2 (assuming 2d in xy-plane)
    delx = x[i2][0] - x[i1][0];
    dely = x[i2][1] - x[i1][1];
    rsq = delx*delx + dely*dely;
    r = sqrt(rsq);
    delx /= r;
    dely /= r;

    if (activestyle==CCW){
      // Apply forces for a net CCW torque.
      f[i1][0] += f_active * (dely);  // unit vector rotated CW
      f[i1][1] += f_active * (-delx);
      f[i2][0] += f_active * (-dely); // unit vector rotated CCW
      f[i2][1] += f_active * (delx);
    }

    else if (activestyle==CW){
      // Apply forces for a net CW torque.
      f[i1][0] += f_active * (-dely); // unit vector rotated CCW
      f[i1][1] += f_active * (delx);
      f[i2][0] += f_active * (dely);  // unit vector rotated CW
      f[i2][1] += f_active * (-delx);
    }

    else if (activestyle==MIXED){
      if (n % 2 == 0){
      // Apply forces for a net CCW torque.
      f[i1][0] += f_active * (dely);  // unit vector rotated CW
      f[i1][1] += f_active * (-delx);
      f[i2][0] += f_active * (-dely); // unit vector rotated CCW
      f[i2][1] += f_active * (delx);
      }
      else{
      // Apply forces for a net CW torque.
      f[i1][0] += f_active * (-dely); // unit vector rotated CCW
      f[i1][1] += f_active * (delx);
      f[i2][0] += f_active * (dely);  // unit vector rotated CW
      f[i2][1] += f_active * (-delx);
      }
    }

    else if (activestyle==CONVECT){
      // Apply convective force along bond axis
      f[i1][0] += f_active * (delx);
      f[i1][1] += f_active * (dely);
      f[i2][0] += f_active * (delx);
      f[i2][1] += f_active * (dely);
    }
  }
}
