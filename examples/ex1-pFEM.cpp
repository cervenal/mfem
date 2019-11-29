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
   int order_high = 3;
   int order_low = 1;
   bool static_cond = false;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order_high, "-o", "--order",
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
   FiniteElementCollection *fec_high;
   FiniteElementCollection *fec_low;
   fec_high = new H1_FECollection(order_high, dim);
   fec_low = new H1_FECollection(order_low, dim);

   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec_high);
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


//   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

   a->Finalize(0);
   int ndofs = fespace->GetNDofs();
   int n_slaves = order_high - order_low;
   int n_true_dofs = ndofs - n_slaves;
   SparseMatrix *cP = new SparseMatrix(ndofs, n_true_dofs);

   Array<int> dofs;
   fespace->GetEdgeDofs(0, dofs);
   const FiniteElement *fe_high = fec_high->FiniteElementForGeometry(Geometry::SEGMENT);
   const FiniteElement *fe_low = fec_low->FiniteElementForGeometry(Geometry::SEGMENT);
   const IntegrationRule &ir = fe_high->GetNodes();


   int n_master = ir.Size() - n_slaves;

   Array<bool> is_true_dof(ndofs);
   is_true_dof = true;

   if (n_slaves > 0)
   {
      DenseMatrix Interpolation;
      IsoparametricTransformation Trans;
      Trans.SetFE(&SegmentFE);
      Trans.SetIdentityTransformation(Geometry::SEGMENT);
      fe_high->GetTransferMatrix(*fe_low, Trans, Interpolation);
      DenseMatrix I_master;
      I_master.CopyRows(Interpolation, 0, n_master-1);
      I_master.Invert();
      DenseMatrix I_slave;
      I_slave.CopyRows(Interpolation, n_master, n_master + n_slaves-1);

      for (int k = 0; k < n_slaves; k++)
      {
         for (int i = 0; i < n_master; i++)
         {
            double coef = 0;
            for (int j = 0; j < n_master; j++)
            {
               coef += I_slave(k,j) * I_master(j,i);
            }
            cP->Add(dofs[n_master+k],dofs[i], coef);
            is_true_dof[dofs[n_master+k]] = false;
         }
      }
   }
   for (int i = 0, true_dof = 0; i < ndofs; i++)
   {
      if (is_true_dof[i]) cP->Add(i, true_dof++, 1.0);
   }


   cP->Finalize();
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
//    a->RecoverFEMSolution(X, *b, x);


   x.SetSize(cP->Height());
   cP->Mult(X, x);
   delete cP;

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
   delete fec_high;
   delete fec_low;
   delete mesh;

   return 0;
}


double solution(const Vector &x)
{
   return (1.0/4.0)*sin(2*M_PI*(sqrt(x(0)*x(0) + x(1)*x(1))));
}
