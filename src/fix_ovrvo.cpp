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
#include "domain.h"
#include "comm.h"
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

  xflag = force->inumeric(FLERR,arg[6]);
  yflag = force->inumeric(FLERR,arg[7]);
  zflag = force->inumeric(FLERR,arg[8]);
  if ((xflag != 0 && xflag != 1) || (yflag != 0 && yflag != 1)
      || (zflag != 0 && zflag != 1))
    error->all(FLERR,"Illegal fix ovrvo command");
  if (zflag && domain->dimension == 2)
    error->all(FLERR,"Fix ovrvo cannot use z dimension for 2d system");

  // allocate the random number generator
  random = new RanPark(lmp,seed + comm->me);
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
  acoeff = exp(-gamma*dt);
  bcoeff = sqrt((2. / (gamma*dt)) * tanh(gamma*dt / 2.));
  vcoeff = sqrt(acoeff);
  ncoeff = sqrt(t_target*(1. - acoeff)); // later modified by mass
}

/* ----------------------------------------------------------------------
   allow for both per-type and per-atom mass
------------------------------------------------------------------------- */

void FixOVRVO::initial_integrate(int /*vflag*/)
{
  // Perform Ornstein-Uehlenbeck velocity randomization (O), then
  // update velocities (V) and update positions (R).

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  double m;
  double sqrtm;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      if (rmass)
        m = rmass[i];
      else
        m = mass[type[i]];
      sqrtm = sqrt(m);

      // Ornstein-Uehlenbeck velocity randomization (O):
      if (xflag)
        v[i][0] = vcoeff * v[i][0] + (ncoeff / sqrtm) * random->gaussian();
      if (yflag)
        v[i][1] = vcoeff * v[i][1] + (ncoeff / sqrtm) * random->gaussian();
      if (zflag)
        v[i][2] = vcoeff * v[i][2] + (ncoeff / sqrtm) * random->gaussian();

      // Velocity update (V):
      if (xflag)
        v[i][0] += bcoeff * dt * f[i][0] / (2. * m);
      else
        v[i][0] += dt * f[i][0] / (2. * m);
      if (yflag)
        v[i][1] += bcoeff * dt * f[i][1] / (2. * m);
      else
        v[i][1] += dt * f[i][1] / (2. * m);
      if (zflag)
        v[i][2] += bcoeff * dt * f[i][2] / (2. * m);
      else
        v[i][2] += dt * f[i][2] / (2. * m);


      // Position update (R):
      if (xflag)
        x[i][0]  += bcoeff * dt * v[i][0];
      else
        x[i][0]  += dt * v[i][0];
      if (yflag)
        x[i][1]  += bcoeff * dt * v[i][1];
      else
        x[i][1]  += dt * v[i][1];
      if (zflag)
        x[i][2]  += bcoeff * dt * v[i][2];
      else
        x[i][2]  += dt * v[i][2];
    }
}

/* ---------------------------------------------------------------------- */

void FixOVRVO::final_integrate()
{
  // Update velocities (V), then perform Ornstein-Uehlenbeck
  // velocity randomization (O).

  double ncoeff_;
  double v_coeff_fm;

  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  double m;
  double sqrtm;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      if (rmass)
        m = rmass[i];
      else
        m = mass[type[i]];
      sqrtm = sqrt(m);

      // Velocity update (V):
      if (xflag)
        v[i][0] += bcoeff * dt * f[i][0] / (2. * m);
      else
        v[i][0] += dt * f[i][0] / (2. * m);
      if (yflag)
        v[i][1] += bcoeff * dt * f[i][1] / (2. * m);
      else
        v[i][1] += dt * f[i][1] / (2. * m);
      if (zflag)
        v[i][2] += bcoeff * dt * f[i][2] / (2. * m);
      else
        v[i][2] += dt * f[i][2] / (2. * m);

      // Ornstein-Uehlenbeck velocity randomization (O):
      if (xflag)
        v[i][0] = vcoeff * v[i][0] + (ncoeff / sqrtm) * random->gaussian();
      if (yflag)
        v[i][1] = vcoeff * v[i][1] + (ncoeff / sqrtm) * random->gaussian();
      if (zflag)
        v[i][2] = vcoeff * v[i][2] + (ncoeff / sqrtm) * random->gaussian();
    }
}

/* ---------------------------------------------------------------------- */

