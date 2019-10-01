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
#include "fix_ovrvo_rotation_com.h"
#include "atom.h"
#include "neighbor.h"
#include "domain.h"
#include "force.h"
#include "comm.h"
#include "update.h"
#include "error.h"
#include "math.h"
#include "random_park.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

// example command
// fix ovrvo/rotation/com TEMP_T TEMP_R GAMMA_T GAMMA_R SEED
FixOVRVORotationCOM::FixOVRVORotationCOM(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (strcmp(style,"ovrvo/rotation/com") != 0 && narg < 6)
    error->all(FLERR,"Illegal fix ovrvo/rotation/com command");
  tt_target = force->numeric(FLERR,arg[3]);
  tr_target = force->numeric(FLERR,arg[4]);
  gamma_t = force->numeric(FLERR,arg[5]);
  gamma_r = force->numeric(FLERR,arg[6]);
  seed = force->inumeric(FLERR,arg[7]);

  // allocate the random number generator
  random = new RanPark(lmp,seed + comm->me);
  time_integrate = 1;

  if (force->newton_bond)
    error->all(FLERR, "To use fix ovrvo/rotation/com, you must turn off newton bonds "
               "in the input file, e.g. with 'newton on off'.");

  if (!comm->ghost_velocity)
    error->all(FLERR, "To use fix ovrvo/rotation/com, you must turn on ghost atom "
               "velocity in the input file with 'comm_modify vel yes'.");

}

/* ---------------------------------------------------------------------- */

int FixOVRVORotationCOM::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixOVRVORotationCOM::init()
{
  dt = update->dt;
  gamma_t4 = gamma_t * dt / 4.;
  gamma_r4 = gamma_r * dt / 4.;
  ncoeff_t = sqrt(tt_target * gamma_t * dt / 8.); // later modified by mass
  ncoeff_r = sqrt(tr_target * gamma_r * dt / 8.); // later modified by mass
}

/* ----------------------------------------------------------------------
   allow for both per-type and per-atom mass
------------------------------------------------------------------------- */

void FixOVRVORotationCOM::initial_integrate(int /*vflag*/)
{
  // Perform Ornstein-Uehlenbeck velocity randomization (O), then
  // update velocities (V) and update positions (R). No timestep rescaling.

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  double m, sqrtm;
  int i1, i2, i3;
  double nx1, nx2, ny1, ny2;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int **anglelist = neighbor->anglelist;
  int nanglelist = neighbor->nanglelist;
  imageint *image = atom->image;
  double remapped_x [3];  // Position array for setting RNG seed
  double vt1 [3];  // Temporary velocity array for atom 1
  double vt2 [3];  // Temporary velocity array for atom 2
  for (int n = 0; n < nanglelist; n++) {
    if (rmass)
      m = rmass[0];
    else
      m = mass[type[0]];
    sqrtm = sqrt(m);

    if (x[anglelist[n][0]][1] < x[anglelist[n][1]][1]) { // Set i1 as left atom
      i1 = anglelist[n][0];
      i2 = anglelist[n][1];
    }
    else {
      i2 = anglelist[n][0];
      i1 = anglelist[n][1];
    }

    for (int j = 0; j < 3; j++) {  // Initialize temporary array of velocities
      vt1[j] = v[i1][j];
      vt2[j] = v[i2][j];
    }

    //Seed RNG using rounded position of leftmost atom in dumbbell
    for (int i = 0; i < 3; i++) remapped_x[i] = x[i1][i]; // Set positions
    domain->remap(remapped_x);                            // Map back into domain
    for (int i = 0; i < 3; i++) {
      remapped_x[i] = (double)round(remapped_x[i] * 10000000)/10000000;  // Round
    }
    random->reset(seed, remapped_x);                      // Seed RNG with position
    nx1 = random->gaussian();
    ny1 = random->gaussian();
    nx2 = random->gaussian();
    ny2 = random->gaussian();

    if (i1 < nlocal){   // Integrate only the real atoms (not ghosts)

      // Ornstein-Uehlenbeck velocity randomization (O):
      v[i1][0] += (-gamma_t4*(vt1[0] + vt2[0])
                  - gamma_r4*(vt1[0] - vt2[0])
                  + (ncoeff_t/sqrtm)*(nx1 + nx2)
                  + (ncoeff_r/sqrtm)*(nx1 - nx2));
      v[i1][1] += (-gamma_t4*(vt1[1] + vt2[1])
                  - gamma_r4*(vt1[1] - vt2[1])
                  + (ncoeff_t/sqrtm)*(ny1 + ny2)
                  + (ncoeff_r/sqrtm)*(ny1 - ny2));
      // Velocity update (V):
      v[i1][0] += dt * f[i1][0] / (2. * m);
      v[i1][1] += dt * f[i1][1] / (2. * m);
      // Position update (R):
      x[i1][0]  += dt * v[i1][0];
      x[i1][1]  += dt * v[i1][1];
    }

    if (i2 < nlocal){   // Integrate only the real atoms (not ghosts)

      // Ornstein-Uehlenbeck velocity randomization (O):
      v[i2][0] += (-gamma_t4*(vt2[0] + vt1[0])
                  - gamma_r4*(vt2[0] - vt1[0])
                  + (ncoeff_t/sqrtm)*(nx2 + nx1)
                  + (ncoeff_r/sqrtm)*(nx2 - nx1));
      v[i2][1] += (-gamma_t4*(vt2[1] + vt1[1])
                  - gamma_r4*(vt2[1] - vt1[1])
                  + (ncoeff_t/sqrtm)*(ny2 + ny1)
                  + (ncoeff_r/sqrtm)*(ny2 - ny1));
      // Velocity update (V):
      v[i2][0] += dt * f[i2][0] / (2. * m);
      v[i2][1] += dt * f[i2][1] / (2. * m);
      // Position update (R):
      x[i2][0]  += dt * v[i2][0];
      x[i2][1]  += dt * v[i2][1];
    }
    i3 = anglelist[n][2];
    if (i3 < nlocal){  // Dummy atom at the COM
      x[i3][0] = 0.5 * (x[i1][0] + x[i2][0]);
      x[i3][1] = 0.5 * (x[i1][1] + x[i2][1]);
      v[i3][0] = 0.5 * (v[i1][0] + v[i2][0]);
      v[i3][1] = 0.5 * (v[i1][1] + v[i2][1]);
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixOVRVORotationCOM::final_integrate()
{
  // Update velocities (V), then perform Ornstein-Uehlenbeck
  // velocity randomization (O).

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  double m, sqrtm;
  int i1, i2, i3;
  double nx1, nx2, ny1, ny2;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int **anglelist = neighbor->anglelist;
  int nanglelist = neighbor->nanglelist;
  imageint *image = atom->image;
  double remapped_x [3];  // Position array for setting RNG seed
  double vt1 [3];  // Temporary velocity array for atom 1
  double vt2 [3];  // Temporary velocity array for atom 2

  for (int n = 0; n < nanglelist; n++) {
    if (rmass)
      m = rmass[0];
    else
      m = mass[type[0]];
    sqrtm = sqrt(m);

    if (x[anglelist[n][0]][1] < x[anglelist[n][1]][1]) { // Set i1 as left atom
      i1 = anglelist[n][0];
      i2 = anglelist[n][1];
    }
    else {
      i2 = anglelist[n][0];
      i1 = anglelist[n][1];
    }

    for (int j = 0; j < 3; j++) {  // Initialize temporary array of velocities
      vt1[j] = v[i1][j];
      vt2[j] = v[i2][j];
    }

    //Seed RNG using rounded position of leftmost atom in dumbbell
    for (int i = 0; i < 3; i++) remapped_x[i] = x[i1][i]; // Set positions
    domain->remap(remapped_x);                            // Map back into domain
    for (int i = 0; i < 3; i++) {
      remapped_x[i] = (double)round(remapped_x[i] * 10000000)/10000000;  // Round
    }
    random->reset(seed, remapped_x);                      // Seed RNG with position
    nx1 = random->gaussian();
    ny1 = random->gaussian();
    nx2 = random->gaussian();
    ny2 = random->gaussian();

    if (i1 < nlocal){   // Integrate only the real atoms (not ghosts)

      // Velocity update (V):
      v[i1][0] += dt * f[i1][0] / (2. * m);
      v[i1][1] += dt * f[i1][1] / (2. * m);
      // Ornstein-Uehlenbeck velocity randomization (O):
      v[i1][0] += (-gamma_t4*(vt1[0] + vt2[0])
                  - gamma_r4*(vt1[0] - vt2[0])
                  + (ncoeff_t/sqrtm)*(nx1 + nx2)
                  + (ncoeff_r/sqrtm)*(nx1 - nx2));
      v[i1][1] += (-gamma_t4*(vt1[1] + vt2[1])
                  - gamma_r4*(vt1[1] - vt2[1])
                  + (ncoeff_t/sqrtm)*(ny1 + ny2)
                  + (ncoeff_r/sqrtm)*(ny1 - ny2));
    }

    if (i2 < nlocal){   // Integrate only the real atoms (not ghosts)

      // Velocity update (V):
      v[i2][0] += dt * f[i2][0] / (2. * m);
      v[i2][1] += dt * f[i2][1] / (2. * m);
      // Ornstein-Uehlenbeck velocity randomization (O):
      v[i2][0] += (-gamma_t4*(vt2[0] + vt1[0])
                  - gamma_r4*(vt2[0] - vt1[0])
                  + (ncoeff_t/sqrtm)*(nx2 + nx1)
                  + (ncoeff_r/sqrtm)*(nx2 - nx1));
      v[i2][1] += (-gamma_t4*(vt2[1] + vt1[1])
                  - gamma_r4*(vt2[1] - vt1[1])
                  + (ncoeff_t/sqrtm)*(ny2 + ny1)
                  + (ncoeff_r/sqrtm)*(ny2 - ny1));
    }
    i3 = anglelist[n][2];
    if (i3 < nlocal){  // Dummy atom at the COM
      x[i3][0] = 0.5 * (x[i1][0] + x[i2][0]);
      x[i3][1] = 0.5 * (x[i1][1] + x[i2][1]);
      v[i3][0] = 0.5 * (v[i1][0] + v[i2][0]);
      v[i3][1] = 0.5 * (v[i1][1] + v[i2][1]);
    }
  }
}

/* ---------------------------------------------------------------------- */

