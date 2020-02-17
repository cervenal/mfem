//                                MFEM Example 1 p-FEM
//
// Sample runs:  ex1-pFEM -o 8 -eo 3
//
// There are 2 quadrilaterals sharing an edge "0"
// The exact solution is projected to an H1 space of order "-o"
// This test shows how to restrict an edge "0" to the prescribed order "-eo"
// by constraining edge functions by themselves.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Exact solution to be projected
double solution(const Vector &x);


// Computes edge self constraint matrix.
// It enforces "edge_order" on the "edge"
// by constraining edge functions by themselves.
SparseMatrix* GetEdgeConstraint(FiniteElementSpace &fespace, int edge, int edge_order);

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/quad-pFEM.mesh";
   int order = 3;
   int edge_order = 2;

   bool static_cond = false;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&edge_order, "-eo", "--edgeorder",
                  "Polynomial order the edge is restricted to");
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

   MFEM_ASSERT(edge_order <= order, "");

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   Array<int> ref;
   ref.SetSize(1);
   ref[0] = 0;
   mesh->GeneralRefinement(ref);

   // 5. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order. If order < 1, we
   //    instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   fec = new H1_FECollection(order, dim);

   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   cout << "Number of finite element unknowns: "
        << fespace->GetTrueVSize() << endl;

   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking all
   //    the boundary attributes from the mesh as essential (Dirichlet) and
   //    converting them to a list of true dofs.
   Array<int> ess_tdof_list;

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
   a->Finalize(0);
   
   OperatorPtr A;
   Vector B, X;

   // Computes constraint matrix.
   //SparseMatrix *cP = GetEdgeConstraint(*fespace, 0, edge_order);
   const SparseMatrix *cP = fespace->GetConformingProlongation();

   SparseMatrix *PT = mfem::Transpose(*cP);
   SparseMatrix *PTA = mfem::Mult(*PT, a->SpMat());
   delete PT;
   A.Reset(mfem::Mult(*PTA, *cP), true);
   delete PTA;

   B.SetSize(cP->Width());
   cP->MultTranspose(*b, B);
   X.SetSize(cP->Width());
   X = 0.0;

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
   x.SetSize(cP->Height());
   cP->Mult(X, x);
   // delete cP;

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
   delete fec;
   delete mesh;

   return 0;
}


SparseMatrix* GetEdgeConstraint(FiniteElementSpace &fespace, int edge, int edge_order)
{

   const FiniteElementCollection* fec = fespace.FEColl();
   const FiniteElement *fe = fec->FiniteElementForGeometry(Geometry::SEGMENT);

   MFEM_ASSERT(edge_order <= fe->GetOrder(), "");

   // New finite element of lower order
   FiniteElementCollection *fec_new = new L2_FECollection(edge_order, fe->GetDim());
   const FiniteElement *fe_new = fec_new->FiniteElementForGeometry(Geometry::SEGMENT);

   // Number of edge DOFs (master and slave)
   int n_master = fe_new->GetDof();
   int n_slaves = fe->GetDof();

   int ndofs = fespace.GetNDofs();

   cout << "cp height" << ndofs << endl;
   cout << "cp width" << ndofs - 2*n_slaves + n_master << endl;

   // Constraint matrix
   SparseMatrix *cP = new SparseMatrix(ndofs, ndofs - 2*n_slaves + n_master);

   Array<int> dofs;
   fespace.GetEdgeDofs(edge, dofs);
   cout << "cp height" << ndofs << endl;
   cout << "cp width" << ndofs - 2*n_slaves + n_master << endl;

   for (int i = 0; i < n_slaves; i++)
   {
     cout << "dofs[i]" << dofs[i] << endl;
   }
   exit(1);

   Array<bool> is_true_dof(ndofs);
   is_true_dof = true;

   // Compute interpolation matrix
   DenseMatrix Interpolation;
   IsoparametricTransformation Trans;
   Trans.SetFE(&SegmentFE);
   Trans.SetIdentityTransformation(Geometry::SEGMENT);
   fe->GetTransferMatrix(*fe_new, Trans, Interpolation);

   for (int i = 0; i < n_slaves; i++)
   {
      for (int j = 0; j < n_master; j++)
      {
         cP->Add(dofs[i], ndofs - 2*n_slaves + j, Interpolation(i,j));
      }
      is_true_dof[dofs[i]] = false;
   }
   // Other dofs are not constrained
   for (int i = 0, true_dof = 0; i < ndofs; i++)
   {
      if (is_true_dof[i])
      {
         cP->Add(i, true_dof++, 1.0);
      }
   }

   cP->Finalize();
   delete fec_new;

   return cP;

}

double solution(const Vector &x)
{
   return sin(2.0*M_PI*(sqrt(x(0)*x(0) + x(1)*x(1))));
}


int bitCount(int n)
{
    int count = 0;
    while (n) {
        n &= (n-1);
        count++;
    };
    return count;
}

