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

Compute the 2x2 stress tensor for a 2D system of actively torqued dumbbells.
The stress tensor is decomposed into four components:
  T_K: The kinetic contribution.
  T_V: The non-bonded virial contribution.
  T_S: The contribution from the dumbbell bond.
  T_A: The active stress, due to a counter-clockwise molecular torque.

The four values of each component are unraveled and appended in the above order
for the returned 16-element vector.

---------------------------------------------------------------------- */


ComputeActiveStress::ComputeActiveStress(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg != 3) error->all(FLERR,"Illegal compute activestress command");

  vector_flag = 1;
  size_vector = 16;      // unraveled 2x2 stress tensors: T^K, T^V, T^S, T^A
  extvector = 0;         // stress tensor vector is intensive, not extensive

  vector = new double[16];
}

/* ---------------------------------------------------------------------- */

ComputeActiveStress::~ComputeActiveStress()
{
  delete [] vector;
}

/* ---------------------------------------------------------------------- */

void ComputeActiveStress::virial_compute()
{
  int i;
  double *T_V, *T_V_all, *T_S, *T_S_all;

  // for (i = 0; i < 4; i++) T_V[i] = 0.0;
  // for (i = 0; i < 4; i++) T_V[i] = 0.0;

  // for (j = 0; j < nvirial; j++) {
  //   vcomponent = vptr[j];
  //   for (i = 0; i < n; i++) v[i] += vcomponent[i];
  // }

  if (force->pair) T_V = force->pair->virial;
  if (force->bond) T_S = force->bond->virial;

  // sum across all processors
  MPI_Allreduce(T_V,T_V_all,4,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(T_S,T_S_all,4,MPI_DOUBLE,MPI_SUM,world);

  for (i = 4; i < 8; i++)
    vector[i] = T_V_all[i];

  for (i = 8; i < 12; i++)
    vector[i] = T_S_all[i];
}


/* ---------------------------------------------------------------------- */

void ComputeActiveStress::kinetic_compute()
{
  int i;
  double **v = atom->v;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  double massone,T_K[4], T_K_all[4];
  for (i = 0; i < 4; i++) T_K[i] = 0.0;

  for (i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      if (rmass) massone = rmass[i];
      else massone = mass[type[i]];
      T_K[0] += massone * v[i][0]*v[i][0];
      T_K[1] += massone * v[i][0]*v[i][1];
      T_K[2] += massone * v[i][1]*v[i][0];
      T_K[3] += massone * v[i][1]*v[i][1];
    }

  // sum across all processors
  MPI_Allreduce(T_K,T_K_all,4,MPI_DOUBLE,MPI_SUM,world);
  for (i = 0; i < 4; i++)
    vector[i] = T_K_all[i] * force->mvv2e;
}



/* ---------------------------------------------------------------------- */


void ComputeActiveStress::compute_vector()
{
  kinetic_compute();
}

/* ---------------------------------------------------------------------- */
