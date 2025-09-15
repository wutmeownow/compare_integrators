/* **********************************************************************
 * Print weights and abscissas for gaussian quadrature rules
 ************************************************************************
 */

#include <stdio.h>
#include <math.h>
#include "integrators.hpp"     


int main() {
  GaussInt gaussint;
  for (int n=2; n<=30; ++n){
    gaussint.Init(n);
    gaussint.PrintWA();
  }
  return 0;
}

  
