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

#include <mpi.h>
#include <cstring>
#include "compute_activestress.h"
#include "atom.h"
#include "neighbor.h"
#include "update.h"
#include "force.h"
#include "pair.h"
#include "bond.h"
#include "domain.h"
#include "comm.h"
#include "group.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ----------------------------------------------------------------------

Compute the 2x2 active stress tensor for a 2D system of torqued dumbbells.
T_A: The active stress, due to a counter-clockwise molecular torque.

If "com" flag is provided, only the center-of-mass velocity of dumbbells
is used in the kinetic part of the stress tensor.

---------------------------------------------------------------------- */


ComputeActiveStress::ComputeActiveStress(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg < 4)
    error->all(FLERR,"Illegal compute activestress command");
  if (narg > 4 && strcmp(arg[4],"com") == 0)
    com=true;
  else
    com=false;
  f_active = force->numeric(FLERR,arg[3]);
  int dimension = domain->dimension;
  if (dimension != 2) error->all(FLERR,"Only 2D stress tensor is supported.");

  extvector = 0;
  timeflag = 1;
  vector_flag = 1;
  size_vector = 4;      // unraveled 2x2 active stress tensor
  extvector = 0;         // stress tensor vector is intensive (divided by vol)

  vector = new double[4];
}

/* ---------------------------------------------------------------------- */

ComputeActiveStress::~ComputeActiveStress()
{
  delete [] vector;
}

/* ---------------------------------------------------------------------- */

// void ComputeActiveStress::kinetic_compute()
// {
//   int i;
//   double vx_com, vy_com;
//   double **v = atom->v;
//   double *mass = atom->mass;
//   double *rmass = atom->rmass;
//   int **bondlist = neighbor->bondlist;
//   int nbondlist = neighbor->nbondlist;
//   int *type = atom->type;
//   int *mask = atom->mask;
//   int nlocal = atom->nlocal;

//   double massone,T_K[4], T_K_all[4];
//   for (i = 0; i < 4; i++) T_K[i] = 0.0;

//   if (com){
//     for (int n = 0; n < nbondlist; n++) {
//       int i1 = bondlist[n][0];
//       int i2 = bondlist[n][1];
//       if (rmass) massone = rmass[i1] + rmass[i2];
//       else massone = mass[type[i1]] + mass[type[i2]];
//       vx_com = (v[i1][0] + v[i2][0]) / 2;  // Assume equal masses
//       vy_com = (v[i1][1] + v[i2][1]) / 2;  // Assume equal masses
//       T_K[0] += massone * vx_com*vx_com;
//       T_K[1] += massone * vx_com*vy_com;
//       T_K[2] += massone * vy_com*vx_com;
//       T_K[3] += massone * vy_com*vy_com;
//     }
//   }
//   else{
//     for (i = 0; i < nlocal; i++)
//       if (mask[i] & groupbit) {
//         if (rmass) massone = rmass[i];
//         else massone = mass[type[i]];
//         T_K[0] += massone * v[i][0]*v[i][0];
//         T_K[1] += massone * v[i][0]*v[i][1];
//         T_K[2] += massone * v[i][1]*v[i][0];
//         T_K[3] += massone * v[i][1]*v[i][1];
//       }
//   }

//   // sum across all processors
//   MPI_Allreduce(T_K,T_K_all,4,MPI_DOUBLE,MPI_SUM,world);
//   for (i = 0; i < 4; i++)
//     vector[i] = T_K_all[i] * force->mvv2e;
// }


// /* ---------------------------------------------------------------------- */

// void ComputeActiveStress::virial_compute()
// {
//   int i;
//   double T_V[4], T_V_all[4], T_S[4], T_S_all[4];
//   double *T_V_ptr, *T_S_ptr;
//   for (i = 0; i < 4; i++)
//     T_V[i] = 0.0;
//     T_S[i] = 0.0;


//   if (force->pair) {
//       T_V_ptr = force->pair->virial;
//       T_V[0] = T_V_ptr[0];
//       T_V[1] = T_V_ptr[3];
//       T_V[2] = T_V_ptr[3];
//       T_V[3] = T_V_ptr[1];
//     }
//   if (force->bond) {
//       T_S_ptr = force->bond->virial;
//       T_S[0] = T_S_ptr[0];
//       T_S[1] = T_S_ptr[3];
//       T_S[2] = T_S_ptr[3];
//       T_S[3] = T_S_ptr[1];
//     }

//   // sum across all processors
//   MPI_Allreduce(T_V,T_V_all,4,MPI_DOUBLE,MPI_SUM,world);
//   MPI_Allreduce(T_S,T_S_all,4,MPI_DOUBLE,MPI_SUM,world);

//   for (i = 0; i < 4; i++)
//     vector[i+4] = T_V_all[i];

//   for (i = 0; i < 4; i++)
//     vector[i+8] = T_S_all[i];
// }

/* ---------------------------------------------------------------------- */

void ComputeActiveStress::active_compute()
{
  int i, n, i1, i2;
  double T_A[4], T_A_all[4];
  double delx, dely, rsq, r;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  int nlocal = atom->nlocal;
  double **x = atom->x;
  for (n = 0; n < 4; n++) T_A[n] = 0.0;
  for (n = 0; n < nbondlist; n++) {
    // int i1 = bondlist[n][0];
    // int i2 = bondlist[n][1];
    if (x[bondlist[n][0]][1] < x[bondlist[n][1]][1]) { // Set i1 as left atom
      i1 = bondlist[n][0];
      i2 = bondlist[n][1];
    }
    else {
      i2 = bondlist[n][0];
      i1 = bondlist[n][1];
    }
    if (i1 < nlocal) {  // Only tally contribution if first atom is not ghost
      // Get a unit vector pointing from atom 1 to atom 2 (assuming 2d in xy-plane)
      delx = x[i2][0] - x[i1][0];
      dely = x[i2][1] - x[i1][1];
      rsq = delx*delx + dely*dely;
      r = sqrt(rsq);

      T_A[0] += f_active * dely * delx / r;
      T_A[1] += f_active * dely * dely / r;
      T_A[2] -= f_active * delx * delx / r;
      T_A[3] -= f_active * delx * dely / r;
    }
  }

  MPI_Allreduce(T_A,T_A_all,4,MPI_DOUBLE,MPI_SUM,world);

  for (i = 0; i < 4; i++)
    vector[i] = T_A_all[i];
}

/* ---------------------------------------------------------------------- */


void ComputeActiveStress::compute_vector()
{
  invoked_vector = update->ntimestep;
  nktv2p = force->nktv2p;
  inv_volume = 1.0 / (domain->xprd * domain->yprd);
  active_compute();
  for (int i = 0; i < 4; i++)
    vector[i] *= inv_volume * nktv2p;
}

/* ---------------------------------------------------------------------- */
