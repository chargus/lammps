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
#include "fix_ovrvo.h"
#include "atom.h"
#include "neighbor.h"
#include "force.h"
#include "update.h"
#include "error.h"
#include "math.h"
#include "random_park.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

// example command
// fix ovrvo TEMP GAMMA SEED
FixOVRVO::FixOVRVO(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (strcmp(style,"ovrvo") != 0 && narg < 6)
    error->all(FLERR,"Illegal fix ovrvo command");
  t_target = force->numeric(FLERR,arg[3]);
  gamma = force->numeric(FLERR,arg[4]);
  int seed = force->inumeric(FLERR,arg[5]);

  // allocate the random number generator
  random = new RanPark(lmp,seed);
  time_integrate = 1;
}

/* ---------------------------------------------------------------------- */

int FixOVRVO::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixOVRVO::init()
{
  dt = update->dt;
  double acoeff = exp(-gamma*dt);
  double bcoeff = sqrt((2. / (gamma*dt)) * tanh(gamma*dt / 2.));

  // Coefficient naming convention: first letter indicates integration step:
  //     o: Ornstein-Uehlenbeck update
  //     v: velocity update
  //     r: position update
  // The second letter indicates which variable coefficient applies to:
  //     v: velocity term
  //     n: random variable term (drawn from standard normal distribution N(0,1))
  //     f: force term
  o_coeff_v = sqrt(acoeff);
  o_coeff_n = sqrt(t_target*(1. - acoeff)); // later modified by mass
  v_coeff_f = bcoeff*dt/2.; // later modified by mass
  r_coeff_v = bcoeff*dt;
}

/* ----------------------------------------------------------------------
   allow for both per-type and per-atom mass
------------------------------------------------------------------------- */

void FixOVRVO::initial_integrate(int /*vflag*/)
{
  // Perform Ornstein-Uehlenbeck velocity randomization (O), then
  // update velocities (V) and update positions (R).

  double o_coeff_nm;
  double v_coeff_fm;

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      if (rmass) {
        o_coeff_nm = o_coeff_n / sqrt(rmass[i]);
        v_coeff_fm = v_coeff_f / rmass[i];
      }
      else {
        o_coeff_nm = o_coeff_n / sqrt(mass[type[i]]);
        v_coeff_fm = v_coeff_f / mass[type[i]];
      }
      // Ornstein-Uehlenbeck velocity randomization (O):
      v[i][0] = o_coeff_v * v[i][0] + o_coeff_nm * random->gaussian();
      v[i][1] = o_coeff_v * v[i][1] + o_coeff_nm * random->gaussian();
      v[i][2] = o_coeff_v * v[i][2] + o_coeff_nm * random->gaussian();

      // Velocity update (V):
      v[i][0] += v_coeff_fm * f[i][0];
      v[i][1] += v_coeff_fm * f[i][1];
      v[i][2] += v_coeff_fm * f[i][2];

      // Position update (R):
      x[i][0]  += r_coeff_v * v[i][0];
      x[i][1]  += r_coeff_v * v[i][1];
      x[i][2]  += r_coeff_v * v[i][2];
    }
}

/* ---------------------------------------------------------------------- */

void FixOVRVO::final_integrate()
{
  // Update velocities (V), then perform Ornstein-Uehlenbeck
  // velocity randomization (O).

  double o_coeff_nm;
  double v_coeff_fm;

  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      if (rmass) {
        o_coeff_nm = o_coeff_n / sqrt(rmass[i]);
        v_coeff_fm = v_coeff_f / rmass[i];
      }
      else {
        o_coeff_nm = o_coeff_n / sqrt(mass[type[i]]);
        v_coeff_fm = v_coeff_f / mass[type[i]];
      }
      // Velocity update (V):
      v[i][0] += v_coeff_fm * f[i][0];
      v[i][1] += v_coeff_fm * f[i][1];
      v[i][2] += v_coeff_fm * f[i][2];

      // Ornstein-Uehlenbeck velocity randomization (O):
      v[i][0] = o_coeff_v * v[i][0] + o_coeff_nm * random->gaussian();
      v[i][1] = o_coeff_v * v[i][1] + o_coeff_nm * random->gaussian();
      v[i][2] = o_coeff_v * v[i][2] + o_coeff_nm * random->gaussian();
    }
}

/* ---------------------------------------------------------------------- */

