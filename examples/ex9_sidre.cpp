//                                MFEM Example 9
//
// Compile with: make ex9
//
// Sample runs:
//    ex9 -m ../data/periodic-segment.mesh -p 0 -r 2 -dt 0.005
//    ex9 -m ../data/periodic-square.mesh -p 0 -r 2 -dt 0.01 -tf 10
//    ex9 -m ../data/periodic-hexagon.mesh -p 0 -r 2 -dt 0.01 -tf 10
//    ex9 -m ../data/periodic-square.mesh -p 1 -r 2 -dt 0.005 -tf 9
//    ex9 -m ../data/periodic-hexagon.mesh -p 1 -r 2 -dt 0.005 -tf 9
//    ex9 -m ../data/amr-quad.mesh -p 1 -r 2 -dt 0.002 -tf 9
//    ex9 -m ../data/star-q3.mesh -p 1 -r 2 -dt 0.005 -tf 9
//    ex9 -m ../data/disc-nurbs.mesh -p 1 -r 3 -dt 0.005 -tf 9
//    ex9 -m ../data/disc-nurbs.mesh -p 2 -r 3 -dt 0.005 -tf 9
//    ex9 -m ../data/periodic-square.mesh -p 3 -r 4 -dt 0.0025 -tf 9 -vs 20
//    ex9 -m ../data/periodic-cube.mesh -p 0 -r 2 -o 2 -dt 0.02 -tf 8
//
// Description:  This example code solves the time-dependent advection equation
//               du/dt + v.grad(u) = 0, where v is a given fluid velocity, and
//               u0(x)=u(0,x) is a given initial condition.
//
//               The example demonstrates the use of Discontinuous Galerkin (DG)
//               bilinear forms in MFEM (face integrators), the use of explicit
//               ODE time integrators, the definition of periodic boundary
//               conditions through periodic meshes, as well as the use of GLVis
//               for persistent visualization of a time-evolving solution. The
//               saving of time-dependent data files for external visualization
//               with VisIt (visit.llnl.gov) is also illustrated.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>

using namespace std;
using namespace mfem;

#include "sidre/sidre.hpp"

// Choice for the problem setup. The fluid velocity, initial condition and
// inflow boundary condition are chosen based on this parameter.
int problem;

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v);

// Initial condition
double u0_function(const Vector &x);

// Inflow boundary condition
double inflow_function(const Vector &x);

// Mesh bounding box
Vector bb_min, bb_max;


/** A time-dependent operator for the right-hand side of the ODE. The DG weak
    form of du/dt = -v.grad(u) is M du/dt = K u + b, where M and K are the mass
    and advection matrices, and b describes the flow on the boundary. This can
    be written as a general ODE, du/dt = M^{-1} (K u + b), and this class is
    used to evaluate the right-hand side. */
class FE_Evolution : public TimeDependentOperator
{
private:
   SparseMatrix &M, &K;
   const Vector &b;
   DSmoother M_prec;
   CGSolver M_solver;

   mutable Vector z;

public:
   FE_Evolution(SparseMatrix &_M, SparseMatrix &_K, const Vector &_b);

   virtual void Mult(const Vector &x, Vector &y) const;

   virtual ~FE_Evolution() { }
};


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   problem = 0;
   const char *mesh_file = "../data/periodic-hexagon.mesh";
   int ref_levels = 2;
   int order = 3;
   int ode_solver_type = 4;
   double t_final = 10.0;
   double dt = 0.01;
   bool visualization = true;
   bool visit = false;
   int vis_steps = 25;

   const char *sidre_restart = "\0";
   bool sidre_use_restart = false;
   const char *sidre_restart_protocol = "conduit_hdf5";

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem setup to use. See options in velocity_function().");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Forward Euler, 2 - RK2 SSP, 3 - RK3 SSP,"
                  " 4 - RK4, 6 - RK6.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&sidre_restart, "-sidre", "--sidre-dump",
                  "Load a sidre dump.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);


   // 1.7 Create datastore
   // see https://rzlc.llnl.gov/confluence/display/MAPP/MFEM+Mesh+Blueprint+Prototype
   std::string fec_type;

   namespace sidre = asctoolkit::sidre;
   sidre::DataStore ds;

   sidre::DataView *elements_connectivity;
   sidre::DataView *material_attribute_values;
   sidre::DataView *coordset_values;


   DataCollection * dc = NULL;
   dc = new SidreDataCollection("Example9", ds.getRoot() );
   sidre::DataGroup* grp = ds.getRoot()->getGroup("Example9");


   if (strcmp(sidre_restart, "\0") != 0) {
      sidre_use_restart = true;
   }

   if (sidre_use_restart) {
      cout << "loading sidre dump (" << sidre_restart_protocol << ") '" 
           << sidre_restart << "'" << endl;

      ds.load(sidre_restart, sidre_restart_protocol);
      fec_type = grp->getView("fields/nodes/basis")->getString();
   }
   else {
      dynamic_cast<SidreDataCollection*>(dc)->SetupMeshBlueprint();

      // Load mesh into a string and save in datastore
      ifstream imesh(mesh_file);
      if (!imesh)
      {
        cerr << "\nCan not open mesh file: " << mesh_file << '\n' << endl;
        return 2;
      }
      std::string meshStr((std::istreambuf_iterator<char>(imesh)),
                           std::istreambuf_iterator<char>());
      imesh.close();

      grp->createViewString("aux/orig_mesh_str", meshStr);
   }

   elements_connectivity = grp->getView("topology/elements/connectivity");
   material_attribute_values = grp->getView("fields/material_attribute/values");


   // 2. Populate the Mesh. Either from the mesh file or from a restart
   Mesh *mesh;

   ElementAllocator *elm_alloc = NULL;
   ElementAllocator *bndry_alloc = NULL;
   Allocator *vertices_alloc = NULL;
   //if (sidre_use_restart) {
   //}
   //else {
      int element_size = 4;

      // Initialize the allocators for the elements in this example
      elm_alloc = new SidreElementAllocator(element_size, elements_connectivity,
        material_attribute_values);
      bndry_alloc = new InternalElementAllocator(8, element_size);
      vertices_alloc = new SidreAllocator<double>(coordset_values);

      // 2. Read the mesh from the given mesh file. We can handle geometrically
      //    periodic meshes in this code.
      std::istringstream istrMesh(grp->getView("aux/orig_mesh_str")->getString() );

      // initialize the mesh from the istringstream
      mesh = new Mesh(istrMesh, elm_alloc, bndry_alloc, vertices_alloc, 1, 1);


      // 4. Refine the mesh to increase the resolution. In this example we do
      //    'ref_levels' of uniform refinement, where 'ref_levels' is a
      //    command-line parameter. If the mesh is of NURBS type, we convert it to
      //    a (piecewise-polynomial) high-order mesh.
      for (int lev = 0; lev < ref_levels; lev++)
      {
         mesh->UniformRefinement();
      }
      if (mesh->NURBSext)
      {
         mesh->SetCurvature(max(order, 1));
      }

      int num_elements = mesh->GetNE();
      elements_connectivity->apply(num_elements, 0, element_size);
      material_attribute_values->apply(num_elements, 0, 1);
   
      {
         Vertex *data = (Vertex*)vertices_alloc->getdata();
         printf("data is %p\n", data);
         int len = vertices_alloc->getcapacity();
         printf ("we own %d vertices\n", len);
         for (int i = 0; i < len; i++) {
            cout << "Vertex " << i << " : " << data[i](1) << " " << data[i](2) << endl;
         }
      }

   //}


   int dim = mesh->Dimension();
   if(! sidre_use_restart)
   {
      std::stringstream sstr;
      sstr << "L2_" << dim << "D_P" << order;
      fec_type = sstr.str();
   }
   else
   {
       fec_type = grp->getView("fields/nodes/basis")->getString();
   }


   // 3. Define the ODE solver used for time integration. Several explicit
   //    Runge-Kutta methods are available.
   ODESolver *ode_solver = NULL;
   switch (ode_solver_type)
   {
      case 1: ode_solver = new ForwardEulerSolver; break;
      case 2: ode_solver = new RK2Solver(1.0); break;
      case 3: ode_solver = new RK3SSPSolver; break;
      case 4: ode_solver = new RK4Solver; break;
      case 6: ode_solver = new RK6Solver; break;
      default:
         cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         return 3;
   }

   // 4. OLD: Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement, where 'ref_levels' is a
   //    command-line parameter. If the mesh is of NURBS type, we convert it to
   //    a (piecewise-polynomial) high-order mesh.
   mesh->GetBoundingBox(bb_min, bb_max, max(order, 1));

   // 5. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   // L2_2D_P3
   //DG_FECollection fec(order, dim);
   //FiniteElementSpace fes(mesh, &fec);
   FiniteElementCollection *fec = FiniteElementCollection::New(fec_type.c_str());
   FiniteElementSpace fes(mesh, fec);

   cout << "the finite element collection has name: '" << fec->Name() << "'" << endl;

   cout << "Number of unknowns: " << fes.GetVSize() << endl;

   // 6. Set up and assemble the bilinear and linear forms corresponding to the
   //    DG discretization. The DGTraceIntegrator involves integrals over mesh
   //    interior faces.
   VectorFunctionCoefficient velocity(dim, velocity_function);
   FunctionCoefficient inflow(inflow_function);
   FunctionCoefficient u0(u0_function);

   BilinearForm m(&fes);
   m.AddDomainIntegrator(new MassIntegrator);
   BilinearForm k(&fes);
   k.AddDomainIntegrator(new ConvectionIntegrator(velocity, -1.0));
   k.AddInteriorFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(velocity, 1.0, -0.5)));
   k.AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(velocity, 1.0, -0.5)));

   LinearForm b(&fes);
   b.AddBdrFaceIntegrator(
      new BoundaryFlowIntegrator(inflow, velocity, -1.0, -0.5));

   m.Assemble();
   m.Finalize();
   int skip_zeros = 0;
   k.Assemble(skip_zeros);
   k.Finalize(skip_zeros);
   b.Assemble();

   // 7. Define the initial conditions, save the corresponding grid function to
   //    a file and (optionally) save data in the VisIt format and initialize
   //    GLVis visualization.
   GridFunction u(&fes, dc->GetFieldData("solution",&fes), fes.GetVSize());
   dc->RegisterField("solution", &u);

   if (!sidre_use_restart) {
      u.ProjectCoefficient(u0);
   }
   else
   {
       mesh->SetNodalGridFunction(&u);
   }

   // test dumping
   if (1)
   {
      dc->Save();
      std::string filename = "ex9_sidre.hdf5";
      std::string protocol = "conduit_hdf5";

      cout << "trying to load '" << filename << "'" << endl;
      asctoolkit::sidre::DataStore copy_ds;
      copy_ds.load(filename, protocol);

      if (ds.getRoot()->isEquivalentTo( copy_ds.getRoot() ) )
      {
        cout << "Datastore save/load with conduit hdf5 passed, they are equivalent." << endl;
      }
      else
      {
       cout << "Datastore conduit hdf5 instances don't match =[" << endl;
       exit(-1);
      }
   }

   {
      ofstream omesh("ex9.mesh");
      omesh.precision(precision);
      mesh->Print(omesh);
      ofstream osol("ex9-init.gf");
      osol.precision(precision);
      u.Save(osol);
   }

   VisItDataCollection visit_dc("Example9", mesh);
   visit_dc.RegisterField("solution", &u);
   if (visit)
   {
      visit_dc.SetCycle(0);
      visit_dc.SetTime(0.0);
      visit_dc.Save();
   }

   socketstream sout;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      sout.open(vishost, visport);
      if (!sout)
      {
         cout << "Unable to connect to GLVis server at "
              << vishost << ':' << visport << endl;
         visualization = false;
         cout << "GLVis visualization disabled.\n";
      }
      else
      {
         sout.precision(precision);
         sout << "solution\n" << *mesh << u;
         sout << "pause\n";
         sout << flush;
         cout << "GLVis visualization paused."
              << " Press space (in the GLVis window) to resume it.\n";
      }
   }

   cout << "number of boundary elements: " << mesh->GetNBE() << endl;

   // 8. Define the time-dependent evolution operator describing the ODE
   //    right-hand side, and perform time-integration (looping over the time
   //    iterations, ti, with a time-step dt).
   FE_Evolution adv(m.SpMat(), k.SpMat(), b);
   ode_solver->Init(adv);

   double t = 0.0;
   for (int ti = 0; true; )
   {
      if (t >= t_final - dt/2)
      {
         break;
      }

      ode_solver->Step(u, t, dt);
      ti++;

      if (ti % vis_steps == 0)
      {
         cout << "time step: " << ti << ", time: " << t << endl;

         if (visualization)
         {
            sout << "solution\n" << *mesh << u << flush;
         }

         if (dc != NULL)
         {
            dc->SetCycle(ti);
            dc->SetTime(t);
            dc->Save();
         }
      }
   }

   // 9. Save the final solution. This output can be viewed later using GLVis:
   //    "glvis -m ex9.mesh -g ex9-final.gf".
   {
      ofstream osol("ex9-final.gf");
      osol.precision(precision);
      u.Save(osol);
   }

   /*
   double *data = u.GetData();
   int u_len = u.Size();
   cout << "GridFunction u's data:" << endl;
   for (int i = 0; i < u_len; i++) {
      cout << data[i] << endl;
   }
   cout << "done" << endl;
   */


   // 10. Free the used memory.
   delete ode_solver;
   delete fec;
   delete mesh;
   delete elm_alloc;
   delete bndry_alloc;
   delete vertices_alloc;

   return 0;
}


// Implementation of class FE_Evolution
FE_Evolution::FE_Evolution(SparseMatrix &_M, SparseMatrix &_K, const Vector &_b)
   : TimeDependentOperator(_M.Size()), M(_M), K(_K), b(_b), z(_M.Size())
{
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(M);

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-9);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);
}

void FE_Evolution::Mult(const Vector &x, Vector &y) const
{
   // y = M^{-1} (K x + b)
   K.Mult(x, z);
   z += b;
   M_solver.Mult(z, y);
}


// Velocity coefficient
void velocity_function(const Vector &x, Vector &v)
{
   int dim = x.Size();

   // map to the reference [-1,1] domain
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = (bb_min[i] + bb_max[i]) * 0.5;
      X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
   }
   switch (problem)
   {
      case 0:
      {
         // Translations in 1D, 2D, and 3D
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = sqrt(2./3.); v(1) = sqrt(1./3.); break;
            case 3: v(0) = sqrt(3./6.); v(1) = sqrt(2./6.); v(2) = sqrt(1./6.);
               break;
         }
         break;
      }
      case 1:
      case 2:
      {
         // Clockwise rotation in 2D around the origin
         const double w = M_PI/2;
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = w*X(1); v(1) = -w*X(0); break;
            case 3: v(0) = w*X(1); v(1) = -w*X(0); v(2) = 0.0; break;
         }
         break;
      }
      case 3:
      {
         // Clockwise twisting rotation in 2D around the origin
         const double w = M_PI/2;
         double d = max((X(0)+1.)*(1.-X(0)),0.) * max((X(1)+1.)*(1.-X(1)),0.);
         d = d*d;
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = d*w*X(1); v(1) = -d*w*X(0); break;
            case 3: v(0) = d*w*X(1); v(1) = -d*w*X(0); v(2) = 0.0; break;
         }
         break;
      }
   }
}

// Initial condition
double u0_function(const Vector &x)
{
   int dim = x.Size();
   // map to the reference [-1,1] domain
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = (bb_min[i] + bb_max[i]) * 0.5;
      X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
   }
   switch (problem)
   {
      case 0:
      case 1:
      {
         switch (dim)
         {
            case 1:
               return exp(-40.*pow(X(0)-0.5,2));
            case 2:
            case 3:
            {
               double rx = 0.45, ry = 0.25, cx = 0., cy = -0.2, w = 10.;
               if (dim == 3)
               {
                  const double s = (1. + 0.25*cos(2*M_PI*X(2)));
                  rx *= s;
                  ry *= s;
               }
               return ( erfc(w*(X(0)-cx-rx))*erfc(-w*(X(0)-cx+rx)) *
                        erfc(w*(X(1)-cy-ry))*erfc(-w*(X(1)-cy+ry)) )/16;
            }
         }
      }
      case 2:
      {
         double x_ = X(0), y_ = X(1), rho, phi;
         rho = hypot(x_, y_)    ;
         phi = atan2(y_, x_);
         return pow(sin(M_PI*rho),2)*sin(3*phi);
      }
      case 3:
      {
         const double f = M_PI;
         return sin(f*X(0))*sin(f*X(1));
      }
   }
   return 0.0;
}

// Inflow boundary condition (zero for the problems considered in this example)
double inflow_function(const Vector &x)
{
   switch (problem)
   {
      case 0:
      case 1:
      case 2:
      case 3: return 0.0;
   }
   return 0.0;
}
