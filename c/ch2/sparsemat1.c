static char help[] = "Assemble a Mat sparsely.\n";

#include <petsc.h>

int main(int argc,char **args) {
  PetscErrorCode ierr;
  Mat        A;
  PetscInt   i, j[4] = {0, 1, 2, 3};
  PetscReal  aA[4][4] = {{ 1.0,  2.0,  3.0,  0.0},
                         { 2.0,  1.0, -2.0, -3.0},
                         {-1.0,  1.0,  1.0,  0.0},
                         { 0.0,  1.0,  1.0, -1.0}};

  ierr = PetscInitialize(&argc,&args,NULL,help); if (ierr) return ierr;

  ierr = MatCreate(PETSC_COMM_WORLD,&A); CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,4,4); CHKERRQ(ierr);
  ierr = MatSetFromOptions(A); CHKERRQ(ierr);
  ierr = MatSetUp(A); CHKERRQ(ierr);
  for (i = 0; i<4; i++){ // set entries of one row at each i
    ierr = MatSetValues(A,1,&i,4,j,aA[i],INSERT_VALUES); CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  MatDestroy(&A);
  return PetscFinalize();
}

