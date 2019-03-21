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
#include "fix_bd.h"
#include "atom.h"
#include "force.h"
#include "update.h"
#include "error.h"
#include "math.h"
#include "random_park.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

// example command
// fix bd TEMP GAMMA SEED
FixBD::FixBD(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (strcmp(style,"bd") != 0 && narg < 6)
    error->all(FLERR,"Illegal fix bd command");
  t_target = force->numeric(FLERR,arg[3]);
  gamma = force->numeric(FLERR,arg[4]);
  int seed = force->inumeric(FLERR,arg[5]);

  // allocate the random number generator
  random = new RanPark(lmp,seed);
  time_integrate = 1;
}

/* ---------------------------------------------------------------------- */

int FixBD::setmask()
{
  int mask = 0;
  mask |= FINAL_INTEGRATE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixBD::init()
{
  dt = update->dt;
}

/* ----------------------------------------------------------------------
   allow for both per-type and per-atom mass
------------------------------------------------------------------------- */

/* ---------------------------------------------------------------------- */

void FixBD::final_integrate()
{

  // friction coefficient, this taken to be a property of the solvent
  // so here gamma_i is gamma / m_i
  double fd_term = 0.;
//  double fd_term_theta = 0.;
  double noise_0,noise_1,noise_2;
//  double noise_theta;
  double **x = atom->x;
  double **f = atom->f;
//  double *theta = atom->theta;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      gamma_i = gamma / mass[type[i]];
      // in LJ units, t_target is given in kbT/epsilon

      fd_term = sqrt(2 * t_target * dt / gamma_i);

      noise_0 = fd_term * random->gaussian();
      noise_1 = fd_term * random->gaussian();
      noise_2 = fd_term * random->gaussian();

      x[i][0] += dt / gamma_i * f[i][0] + noise_0;
      x[i][1] += dt / gamma_i * f[i][1] + noise_1;
      x[i][2] += dt / gamma_i * f[i][2] + noise_2;

    }
  }
}

/* ---------------------------------------------------------------------- */

