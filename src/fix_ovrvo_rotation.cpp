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
#include "comm.h"
#include "update.h"
#include "error.h"
#include "math.h"
#include "random_park.h"
#include <iostream>

using namespace std;
using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

// example command
// fix ovrvo/rotation TEMP_T TEMP_R GAMMA_T GAMMA_R SEED
FixOVRVORotation::FixOVRVORotation(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (strcmp(style,"ovrvo/rotation") != 0 && narg < 6)
    error->all(FLERR,"Illegal fix ovrvo/rotation command");
  tt_target = force->numeric(FLERR,arg[3]);
  tr_target = force->numeric(FLERR,arg[4]);
  gamma_t = force->numeric(FLERR,arg[5]);
  gamma_r = force->numeric(FLERR,arg[6]);
  seed = force->inumeric(FLERR,arg[7]);

  // allocate the random number generator
  random = new RanPark(lmp,seed + comm->me);
  time_integrate = 1;

  if (force->newton_bond)
    error->all(FLERR, "To use fix ovrvo/rotation, you must turn off newton bonds "
               "in the input file, e.g. with 'newton on off'.");
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
  int i1, i2;
  double nx1, nx2, ny1, ny2;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  imageint *image = atom->image;
  double remapped_x [3];

  for (int n = 0; n < nbondlist; n++) {
    // int i1 = bondlist[n][0];
    // int i2 = bondlist[n][1];

    if (rmass)
      m = rmass[n];
    else
      m = mass[type[n]];
    sqrtm = sqrt(m);

    if (x[bondlist[n][0]][0] < x[bondlist[n][1]][0]) { // Set i1 as left atom
      // cout << "IN if statement  " << x[bondlist[n][0]][0] << "  " << x[bondlist[n][1]][0] << endl;
      i1 = bondlist[n][0];
      i2 = bondlist[n][1];
    }
    else {
      // cout << "IN else statement  " << x[bondlist[n][0]][0] << "  " << x[bondlist[n][1]][0] << endl;
      i2 = bondlist[n][0];
      i1 = bondlist[n][1];
    }
    // float floatseed
    // int seed = (int)abs(1000000*(x[i1][0]-x[i2][0]));
    // cout << "seed:  " << seed << endl;
    // for (i = 0; i < 5; i++) random->uniform(); // warm up the RNG
    // double dx [3];
    // for (int i = 0; i < 3; i++) {
    //   dx[i] = abs(x[i2][i]-x[i1][i]);
    //   cout << abs(x[i2][i]-x[i1][i]) << endl;
    //   cout << dx[i] << endl;}
    // random->reset(seed, x[i1]);
    // cout << "oldx, newx:  " << x[i2][0] << "  " << remapped_x[0] << endl;
    // if (x[i1][0] < x[i2][0]) {
      for (int i = 0; i < 3; i++) remapped_x[i] = x[i1][i];
      domain->remap(remapped_x, image[i1]);
      random->reset(seed, remapped_x);  // Use left atom position to set seed
      nx1 = random->gaussian();
      ny1 = random->gaussian();
      nx2 = random->gaussian();
      ny2 = random->gaussian();
    // }
    // else {
    //   for (int i = 0; i < 3; i++) remapped_x[i] = x[i2][i];
    //   domain->remap(remapped_x, image[i2]);
    //   random->reset(seed, remapped_x);  // Use left atom position to set seed
    //   nx2 = random->gaussian();
    //   ny2 = random->gaussian();
    //   nx1 = random->gaussian();
    //   ny1 = random->gaussian();
    // }
      // cout << n << "  " << i1 << "  " << x[i1][0] << "  " << remapped_x[0] << "  " << nx1 << endl;
      // cout << n << "  " << i2 << "  " << x[i2][0] << "  " << remapped_x[0] << "  " << nx2 << endl;


    // cout << update->ntimestep << "  " << n << "  " << i1 << "  " << x[i1][0] << "  " <<
    // - gamma_r4*(v[i1][0] - v[i2][0]) << "  " << v[i1][0] << "  " << v[i2][0] << endl;

    // cout << update->ntimestep << "  " << n << "  " << i2 << "  " << x[i2][0] << "  " <<
    // - gamma_r4*(v[i2][0] - v[i1][0]) << "  " << v[i2][0] << "  " << v[i1][0] << endl;
    if (i1 < nlocal){   // Integrate only the real atoms (not ghosts)

      // Ornstein-Uehlenbeck velocity randomization (O):

      v[i1][0] += (-gamma_t4*(v[i1][0] + v[i2][0])
                  - gamma_r4*(v[i1][0] - v[i2][0])
                  + ncoeff_t*(nx1 + nx2)
                  + ncoeff_r*(nx1 - nx2));
      v[i1][1] += (-gamma_t4*(v[i1][1] + v[i2][1])
                  - gamma_r4*(v[i1][1] - v[i2][1])
                  + ncoeff_t*(ny1 + ny2)
                  + ncoeff_r*(ny1 - ny2));
      // Velocity update (V):
      v[i1][0] += dt * f[i1][0] / (2. * m);
      v[i1][1] += dt * f[i1][1] / (2. * m);
      // Position update (R):
      x[i1][0]  += dt * v[i1][0];
      x[i1][1]  += dt * v[i1][1];
    }

    if (i2 < nlocal){   // Integrate only the real atoms (not ghosts)

      // Ornstein-Uehlenbeck velocity randomization (O):
      v[i2][0] += (-gamma_t4*(v[i2][0] + v[i1][0])
                  - gamma_r4*(v[i2][0] - v[i1][0])
                  + ncoeff_t*(nx2 + nx1)
                  + ncoeff_r*(nx2 - nx1));
      v[i2][1] += (-gamma_t4*(v[i2][1] + v[i1][1])
                  - gamma_r4*(v[i2][1] - v[i1][1])
                  + ncoeff_t*(ny2 + ny1)
                  + ncoeff_r*(ny2 - ny1));
      // Velocity update (V):
      v[i2][0] += dt * f[i2][0] / (2. * m);
      v[i2][1] += dt * f[i2][1] / (2. * m);
      // Position update (R):
      x[i2][0]  += dt * v[i2][0];
      x[i2][1]  += dt * v[i2][1];
    }
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
  int i1, i2;
  double nx1, nx2, ny1, ny2;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  imageint *image = atom->image;
  double remapped_x [3];

  for (int n = 0; n < nbondlist; n++) {
    // int i1 = bondlist[n][0];
    // int i2 = bondlist[n][1];

    if (rmass)
      m = rmass[n];
    else
      m = mass[type[n]];
    sqrtm = sqrt(m);

    if (x[bondlist[n][0]][0] < x[bondlist[n][1]][0]) { // Set i1 as left atom
      i1 = bondlist[n][0];
      i2 = bondlist[n][1];
    }
    else {
      i2 = bondlist[n][0];
      i1 = bondlist[n][1];
    }
    // if (x[i1][0] < x[i2][0]) {
      for (int i = 0; i < 3; i++) remapped_x[i] = x[i1][i];
      domain->remap(remapped_x, image[i1]);
      random->reset(seed, remapped_x);  // Use left atom position to set seed
      nx1 = random->gaussian();
      ny1 = random->gaussian();
      nx2 = random->gaussian();
      ny2 = random->gaussian();
    // }
    // else {
    //   for (int i = 0; i < 3; i++) remapped_x[i] = x[i2][i];
    //   domain->remap(remapped_x, image[i2]);
    //   random->reset(seed, remapped_x);  // Use left atom position to set seed
    //   nx2 = random->gaussian();
    //   ny2 = random->gaussian();
    //   nx1 = random->gaussian();
    //   ny1 = random->gaussian();
    // }

    if (i1 < nlocal){   // Integrate only the real atoms (not ghosts)

      // Velocity update (V):
      v[i1][0] += dt * f[i1][0] / (2. * m);
      v[i1][1] += dt * f[i1][1] / (2. * m);
      // Ornstein-Uehlenbeck velocity randomization (O):
      v[i1][0] += (-gamma_t4*(v[i1][0] + v[i2][0])
                  - gamma_r4*(v[i1][0] - v[i2][0])
                  + ncoeff_t*(nx1 + nx2)
                  + ncoeff_r*(nx1 - nx2));
      v[i1][1] += (-gamma_t4*(v[i1][1] + v[i2][1])
                  - gamma_r4*(v[i1][1] - v[i2][1])
                  + ncoeff_t*(ny1 + ny2)
                  + ncoeff_r*(ny1 - ny2));
    }
    if (i2 < nlocal){   // Integrate only the real atoms (not ghosts)

      // Velocity update (V):
      v[i2][0] += dt * f[i2][0] / (2. * m);
      v[i2][1] += dt * f[i2][1] / (2. * m);
      // Ornstein-Uehlenbeck velocity randomization (O):
      v[i2][0] += (-gamma_t4*(v[i2][0] + v[i1][0])
                  - gamma_r4*(v[i2][0] - v[i1][0])
                  + ncoeff_t*(nx2 + nx1)
                  + ncoeff_r*(nx2 - nx1));
      v[i2][1] += (-gamma_t4*(v[i2][1] + v[i1][1])
                  - gamma_r4*(v[i2][1] - v[i1][1])
                  + ncoeff_t*(ny2 + ny1)
                  + ncoeff_r*(ny2 - ny1));
    }
  }
}

/* ---------------------------------------------------------------------- */

