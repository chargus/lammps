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
#include "force.h"
#include "update.h"
#include "error.h"
#include "math.h"
#include "random_park.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

// example command
// fix dumbbell TEMP GAMMA SEED
FixDumbbell::FixDumbbell(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (strcmp(style,"dumbbell") != 0 && narg < 6)
    error->all(FLERR,"Illegal fix dumbbell command");
  t_target = force->numeric(FLERR,arg[3]);
  gamma = force->numeric(FLERR,arg[4]);
  int seed = force->inumeric(FLERR,arg[5]);

  // allocate the random number generator
  random = new RanPark(lmp,seed);
  time_integrate = 1;
}

/* ---------------------------------------------------------------------- */

int FixDumbbell::setmask()
{
  int mask = 0;
  mask |= FINAL_INTEGRATE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixDumbbell::init()
{
  dt = update->dt;
}

/* ----------------------------------------------------------------------
   allow for both per-type and per-atom mass
------------------------------------------------------------------------- */

/* ---------------------------------------------------------------------- */

void FixDumbbell::final_integrate()
{

  // friction coefficient, this taken to be a property of the solvent
  // so here gamma_i is gamma / m_i
  double fd_term = 0.;
  double noise_0,noise_1,noise_2;
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      // in LJ units, t_target is given in kbT/epsilon
      fd_term = sqrt(2 * t_target * gamma * dt / mass[type[i]]);

      noise_0 = fd_term * random->gaussian();
      noise_1 = fd_term * random->gaussian();
      noise_2 = fd_term * random->gaussian();

      x[i][0] += v[i][0] * dt;
      x[i][1] += v[i][1] * dt;
      x[i][2] += v[i][2] * dt;

      v[i][0] += (f[i][0] * dt / mass[type[i]]) - (gamma * v[i][0] * dt) + noise_0;
      v[i][1] += (f[i][1] * dt / mass[type[i]]) - (gamma * v[i][1] * dt) + noise_1;
      v[i][2] += (f[i][2] * dt / mass[type[i]]) - (gamma * v[i][2] * dt) + noise_2;

    }
  }
}

/* ---------------------------------------------------------------------- */

