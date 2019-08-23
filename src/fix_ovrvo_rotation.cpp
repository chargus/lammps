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
#include "fix_ovrvo_rotation.h"
#include "atom.h"
#include "neighbor.h"
#include "domain.h"
#include "force.h"
#include "update.h"
#include "error.h"
#include "math.h"
#include "random_park.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

// example command
// fix ovrvo/rotation TEMP GAMMA_T GAMMA_R SEED
FixOVRVORotation::FixOVRVORotation(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (strcmp(style,"ovrvo/rotation") != 0 && narg < 6)
    error->all(FLERR,"Illegal fix ovrvo/rotation command");
  tt_target = force->numeric(FLERR,arg[3]);
  tr_target = force->numeric(FLERR,arg[4]);
  gamma_t = force->numeric(FLERR,arg[5]);
  gamma_r = force->numeric(FLERR,arg[6]);
  int seed = force->inumeric(FLERR,arg[7]);

  // allocate the random number generator
  random = new RanPark(lmp,seed);
  time_integrate = 1;
}

/* ---------------------------------------------------------------------- */

int FixOVRVORotation::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixOVRVORotation::init()
{
  dt = update->dt;
  gamma_t4 = gamma_t * dt / 4.;
  gamma_r4 = gamma_r * dt / 4.;
  ncoeff_t = sqrt(tt_target * gamma_t * dt) / 2.; // later modified by mass
  ncoeff_r = sqrt(tr_target * gamma_r * dt) / 2.; // later modified by mass
}

/* ----------------------------------------------------------------------
   allow for both per-type and per-atom mass
------------------------------------------------------------------------- */

void FixOVRVORotation::initial_integrate(int /*vflag*/)
{
  // Perform Ornstein-Uehlenbeck velocity randomization (O), then
  // update velocities (V) and update positions (R). No timestep rescaling.

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  double m, sqrtm;
  double nx1, nx2, ny1, ny2, nz1, nz2;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  for (int i = 0; i < nbondlist; i++) {
    int i1 = bondlist[i][0];
    int i2 = bondlist[i][1];

    if (rmass)
      m = rmass[i];
    else
      m = mass[type[i]];
    sqrtm = sqrt(m);

    // Ornstein-Uehlenbeck velocity randomization (O):
    nx1 = random->gaussian();
    nx2 = random->gaussian();
    ny1 = random->gaussian();
    ny2 = random->gaussian();
    nz1 = random->gaussian();
    nz2 = random->gaussian();

    v[i1][0] += (-gamma_t4*(v[i1][0] + v[i2][0])
                - gamma_r4*(v[i1][0] - v[i2][0])
                + ncoeff_t*(nx1 + nx2)
                + ncoeff_r*(nx1 - nx2));
    v[i2][0] += (-gamma_t4*(v[i2][0] + v[i1][0])
                - gamma_r4*(v[i2][0] - v[i1][0])
                + ncoeff_t*(nx2 + nx1)
                + ncoeff_r*(nx2 - nx1));

    v[i1][1] += (-gamma_t4*(v[i1][1] + v[i2][1])
                - gamma_r4*(v[i1][1] - v[i2][1])
                + ncoeff_t*(ny1 + ny2)
                + ncoeff_r*(ny1 - ny2));
    v[i2][1] += (-gamma_t4*(v[i2][1] + v[i1][1])
                - gamma_r4*(v[i2][1] - v[i1][1])
                + ncoeff_t*(ny2 + ny1)
                + ncoeff_r*(ny2 - ny1));

    v[i1][2] += (-gamma_t4*(v[i1][2] + v[i2][2])
                - gamma_r4*(v[i1][2] - v[i2][2])
                + ncoeff_t*(nz1 + nz2)
                + ncoeff_r*(nz1 - nz2));
    v[i2][2] += (-gamma_t4*(v[i2][2] + v[i1][2])
                - gamma_r4*(v[i2][2] - v[i1][2])
                + ncoeff_t*(nz2 + nz1)
                + ncoeff_r*(nz2 - nz1));

    // Velocity update (V):
    v[i1][0] += dt * f[i1][0] / (2. * m);
    v[i2][0] += dt * f[i2][0] / (2. * m);
    v[i1][1] += dt * f[i1][1] / (2. * m);
    v[i2][1] += dt * f[i2][1] / (2. * m);
    v[i1][2] += dt * f[i1][2] / (2. * m);
    v[i2][2] += dt * f[i2][2] / (2. * m);

    // Position update (R):
    x[i1][0]  += dt * v[i1][0];
    x[i2][0]  += dt * v[i2][0];
    x[i1][1]  += dt * v[i1][1];
    x[i2][1]  += dt * v[i2][1];
    x[i1][2]  += dt * v[i1][2];
    x[i2][2]  += dt * v[i2][2];
  }
}

/* ---------------------------------------------------------------------- */

void FixOVRVORotation::final_integrate()
{
  // Update velocities (V), then perform Ornstein-Uehlenbeck
  // velocity randomization (O).

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  double m, sqrtm;
  double nx1, nx2, ny1, ny2, nz1, nz2;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  for (int i = 0; i < nbondlist; i++) {
    int i1 = bondlist[i][0];
    int i2 = bondlist[i][1];

    if (rmass)
      m = rmass[i];
    else
      m = mass[type[i]];
    sqrtm = sqrt(m);

    // Velocity update (V):
    v[i1][0] += dt * f[i1][0] / (2. * m);
    v[i2][0] += dt * f[i2][0] / (2. * m);
    v[i1][1] += dt * f[i1][1] / (2. * m);
    v[i2][1] += dt * f[i2][1] / (2. * m);
    v[i1][2] += dt * f[i1][2] / (2. * m);
    v[i2][2] += dt * f[i2][2] / (2. * m);

    // Ornstein-Uehlenbeck velocity randomization (O):
    nx1 = random->gaussian();
    nx2 = random->gaussian();
    ny1 = random->gaussian();
    ny2 = random->gaussian();
    nz1 = random->gaussian();
    nz2 = random->gaussian();

    v[i1][0] += (-gamma_t4*(v[i1][0] + v[i2][0])
                - gamma_r4*(v[i1][0] - v[i2][0])
                + ncoeff_t*(nx1 + nx2)
                + ncoeff_r*(nx1 - nx2));
    v[i2][0] += (-gamma_t4*(v[i2][0] + v[i1][0])
                - gamma_r4*(v[i2][0] - v[i1][0])
                + ncoeff_t*(nx2 + nx1)
                + ncoeff_r*(nx2 - nx1));

    v[i1][1] += (-gamma_t4*(v[i1][1] + v[i2][1])
                - gamma_r4*(v[i1][1] - v[i2][1])
                + ncoeff_t*(ny1 + ny2)
                + ncoeff_r*(ny1 - ny2));
    v[i2][1] += (-gamma_t4*(v[i2][1] + v[i1][1])
                - gamma_r4*(v[i2][1] - v[i1][1])
                + ncoeff_t*(ny2 + ny1)
                + ncoeff_r*(ny2 - ny1));

    v[i1][2] += (-gamma_t4*(v[i1][2] + v[i2][2])
                - gamma_r4*(v[i1][2] - v[i2][2])
                + ncoeff_t*(nz1 + nz2)
                + ncoeff_r*(nz1 - nz2));
    v[i2][2] += (-gamma_t4*(v[i2][2] + v[i1][2])
                - gamma_r4*(v[i2][2] - v[i1][2])
                + ncoeff_t*(nz2 + nz1)
                + ncoeff_r*(nz2 - nz1));
  }
}

/* ---------------------------------------------------------------------- */

