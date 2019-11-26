//                                MFEM Example 1
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double solution(const Vector &x);

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/quad-pFEM.mesh";
   int order =3;
   bool static_cond = false;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 50,000
   //    elements.
   {
      int ref_levels = 0;
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 5. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order. If order < 1, we
   //    instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
   }
   else if (mesh->GetNodes())
   {
      fec = mesh->GetNodes()->OwnFEC();
      cout << "Using isoparametric FEs: " << fec->Name() << endl;
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
   }
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   cout << "Number of finite element unknowns: "
        << fespace->GetTrueVSize() << endl;

   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking all
   //    the boundary attributes from the mesh as essential (Dirichlet) and
   //    converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   /*if (mesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }*/

   // 7. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (solution,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   LinearForm *b = new LinearForm(fespace);
   FunctionCoefficient rhs(solution);
   b->AddDomainIntegrator(new DomainLFIntegrator(rhs));
   b->Assemble();

   // 8. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x(fespace);
   x = 0.0;

   // 9. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Mass operator (u,v)
   BilinearForm *a = new BilinearForm(fespace);
   ConstantCoefficient one(1.0);
   if (pa) { a->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a->AddDomainIntegrator(new MassIntegrator(one));

   // 10. Assemble the bilinear form and the corresponding linear system,
   //     applying any necessary transformations such as: eliminating boundary
   //     conditions, applying conforming constraints for non-conforming AMR,
   //     static condensation, etc.
   if (static_cond) { a->EnableStaticCondensation(); }
   a->Assemble();

   OperatorPtr A;
   Vector B, X;


   //a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

   a->Finalize(0);
   int ndofs = fespace->GetNDofs();
   int n_true_dofs = ndofs-1;
   SparseMatrix *cP = new SparseMatrix(ndofs, n_true_dofs);

   Array<int> dofs;
   fespace->GetEdgeDofs(0, dofs);
   const FiniteElement *fe = fec->FiniteElementForGeometry(Geometry::SEGMENT);
   const IntegrationRule &ir = fe->GetNodes();

   ir.Size();
   cout << "Nodes: " << ir[0].x << endl;
   cout << "Nodes: " << ir[1].x << endl;
   cout << "Nodes: " << ir[2].x << endl;
   cout << "Nodes: " << ir[3].x << endl;

//   double coef1 = -1.0/9.0 * 1.0 + 8.0/9.0 * -1.0/4.0 + 2.0/9.0 * 0.0;
//   double coef2 = -1.0/9.0 * 0.0 + 8.0/9.0 *  9.0/8.0 + 2.0/9.0 * 0.0;
//   double coef4 = -1.0/9.0 * 0.0 + 8.0/9.0 *  1.0/8.0 + 2.0/9.0 * 1.0;
//   cP->Add(dofs[3],dofs[0], coef1);
//   cP->Add(dofs[3],dofs[1], coef4);
//   cP->Add(dofs[3],dofs[2], coef2);

//   for (int i = 0, true_dof = 0; i < ndofs; i++)
//   {
//      if (i != dofs[3]) cP->Add(i, true_dof++, 1.0);
//   }

     double a2 = (1-ir[2].x) * 1.0 + ir[2].x * 0.0;
     double b2 = (1-ir[2].x) * 0.0 + ir[2].x * 1.0;
     double a3 = (1-ir[3].x) * 1.0 + ir[3].x * 0.0;
     double b3 = (1-ir[3].x) * 0.0 + ir[3].x * 1.0;
     cP->Add(dofs[2],dofs[0], a2);
     cP->Add(dofs[2],dofs[1], b2);
     cP->Add(dofs[3],dofs[0], a3);
     cP->Add(dofs[3],dofs[1], b3);

      for (int i = 0, true_dof = 0; i < ndofs; i++)
      {
         if ((i != dofs[2]) && (i != dofs[3])) cP->Add(i, true_dof++, 1.0);
         //cP->Add(i, true_dof++, 1.0);
      }


   cP->Finalize();
   cP->PrintMatlab();
   SparseMatrix *PT = Transpose(*cP);
   SparseMatrix *PTA = mfem::Mult(*PT, a->SpMat());
   delete PT;
   A.Reset(mfem::Mult(*PTA, *cP), false);
   delete PTA;

   B.SetSize(cP->Width());
   cP->MultTranspose(*b, B);

   X.SetSize(n_true_dofs);









   cout << "Size of linear system: " << A->Height() << endl;

   // 11. Solve the linear system A X = B.
   if (!pa)
   {
#ifndef MFEM_USE_SUITESPARSE
      // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
      GSSmoother M((SparseMatrix&)(*A));
      PCG(*A, M, B, X, 1, 200, 1e-12, 0.0);
#else
      // If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
      UMFPackSolver umf_solver;
      umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
      umf_solver.SetOperator(*A);
      umf_solver.Mult(B, X);
#endif
   }
   else // No preconditioning for now in partial assembly mode.
   {
      CG(*A, B, X, 1, 2000, 1e-12, 0.0);
   }

   // 12. Recover the solution as a finite element grid function.
   // a->RecoverFEMSolution(X, *b, x);


   x.SetSize(cP->Height());
   cP->Mult(X, x);
   delete cP;
   x.Print(cout,1);

   // 13. Save the refined mesh and the solution. This output can be viewed later
   //     using GLVis: "glvis -m refined.mesh -g sol.gf".
   ofstream mesh_ofs("refined.mesh");
   mesh_ofs.precision(8);
   mesh->Print(mesh_ofs);
   ofstream sol_ofs("sol.gf");
   sol_ofs.precision(8);
   x.Save(sol_ofs);

   // 14. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << x << flush;
   }

   // 15. Free the used memory.
   delete a;
   delete b;
   delete fespace;
   if (order > 0) { delete fec; }
   delete mesh;

   return 0;
}


double solution(const Vector &x)
{
   return (1.0/4.0)*sin(2*M_PI*(sqrt(x(0)*x(0) + x(1)*x(1))));
}
