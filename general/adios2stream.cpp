/*
 * Distributed under the OSI-approved Apache License, Version 2.0.  See
 * accompanying file Copyright.txt for details.
 *
 * adios2stream.cpp : implementation of adios2stream functions
 *
 *  Created on: Feb 4, 2019
 *      Author: William F Godoy godoywf@ornl.gov
 */

#include "adios2stream.hpp"

#include "../fem/geom.hpp"
#include "../general/array.hpp"
#include "../mesh/element.hpp"
#include "../mesh/mesh.hpp"
#include "../fem/gridfunc.hpp"


#ifdef MFEM_USE_MPI
#include "../mesh/pmesh.hpp"
#include "../fem/pgridfunc.hpp"
#endif

namespace mfem
{

const std::string adios2stream::vtk_schema_pre = R"(
<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
  <UnstructuredGrid>
    <Piece NumberOfPoints="NumOfVertices" NumberOfCells="NumOfElements">   
      <Points>
        <DataArray Name="vertices" />
      </Points>
      <Cells>
        <DataArray Name="connectivity" />
        <DataArray Name="types" />
      </Cells>)";

const std::string adios2stream::vtk_schema_post = R"(
    </Piece>
  </UnstructuredGrid>
</VTKFile>)";

namespace
{
/**
 * convert openmode input to adios2::Mode format for adios2openmode placeholder
 * @param mode input
 * @return adios2::Mode format
 */
adios2::Mode ToADIOS2Mode(const adios2stream::openmode mode)
{
   adios2::Mode adios2Mode = adios2::Mode::Undefined;
   switch (mode)
   {
      case adios2stream::openmode::out:
         adios2Mode = adios2::Mode::Write;
         break;
      case adios2stream::openmode::in:
         adios2Mode = adios2::Mode::Read;
         break;
      default:
         throw std::invalid_argument(
            "ERROR: invalid adios2stream, "
            "only openmode::out and "
            "openmode::in are valid, "
            "in call to adios2stream constructor\n");
   }
   return adios2Mode;
}

}  // namespace

// PUBLIC
#ifdef MFEM_USE_MPI
adios2stream::adios2stream(const std::string& name, const openmode mode,
                           MPI_Comm comm, const std::string engineType)
   : name(name),
     adios2_openmode(mode),
     adios(std::make_shared<adios2::ADIOS>(comm)),
     io(adios->DeclareIO(name))
{
   io.SetEngine(engineType);
}
#else
adios2stream::adios2stream(const std::string& name, const openmode mode,
                           const std::string engineType)
   : name(name),
     adios2_openmode(mode),
     adios(std::make_shared<adios2::ADIOS>()),
     io(adios->DeclareIO(name))
{
   io.SetEngine(engineType);
}
#endif

adios2stream::~adios2stream()
{
   if (engine)
   {
      adios2::Attribute<std::string> vtkSchema =
         io.InquireAttribute<std::string>("vtk.xml");
      if (!vtkSchema)
      {
         io.DefineAttribute<std::string>("vtk.xml", VTKSchema() );
      }
      engine.Close();
   }
}

void adios2stream::SetParameters(
   const std::map<std::string, std::string>& parameters)
{
   io.SetParameters(parameters);
}

void adios2stream::SetParameter(const std::string key,
                                const std::string value)
{
   io.SetParameter(key, value);
}

void adios2stream::BeginStep()
{
   if (!engine)
   {
      engine = io.Open(name, adios2::Mode::Write);
   }
   engine.BeginStep();
   active_step = true;
}

void adios2stream::EndStep()
{
   if (!engine || active_step == false)
   {
      throw std::logic_error("MFEM adios2stream: calling EndStep on an uninitialized step with BeginStep\n");
   }

   adios2::Attribute<std::string> vtkSchema =
      io.InquireAttribute<std::string>("vtk.xml");
   if (!vtkSchema)
   {
      io.DefineAttribute<std::string>("vtk.xml", VTKSchema() );
   }

   engine.EndStep();
   active_step = false;
}

size_t adios2stream::CurrentStep() const noexcept
{
   return engine.CurrentStep();
}

void adios2stream::Close()
{
   if (engine)
   {
      adios2::Attribute<std::string> vtkSchema =
         io.InquireAttribute<std::string>("vtk.xml");
      if (!vtkSchema)
      {
         io.DefineAttribute<std::string>("vtk.xml", VTKSchema() );
      }
      engine.Close();
   }
   if (adios)
   {
      adios.reset();
   }
}


// PROTECTED (accessible by friend classes)
int32_t adios2stream::GLVISToVTKType(
   const int glvisType) const noexcept
{
   uint32_t vtkType = 0;
   switch (glvisType)
   {
      case Geometry::Type::POINT:
         vtkType = 1;
         break;
      case Geometry::Type::SEGMENT:
         vtkType = 3;
         break;
      case Geometry::Type::TRIANGLE:
         vtkType = 5;
         break;
      case Geometry::Type::SQUARE:
         vtkType = 8;
         //vtkType = 9;
         break;
      case Geometry::Type::TETRAHEDRON:
         vtkType = 10;
         break;
      case Geometry::Type::CUBE:
         vtkType = 11;
         break;
      case Geometry::Type::PRISM:
         vtkType = 13;
         break;
      default:
         vtkType = 0;
         break;
   }
   return vtkType;
}

bool adios2stream::IsConstantElementType(const Array<Element*>& elements ) const
noexcept
{
   bool isConstType = true;
   const Geometry::Type type = elements[0]->GetGeometryType();

   for (int e = 1; e < elements.Size(); ++e)
   {
      if (type != elements[e]->GetGeometryType())
      {
         isConstType = false;
         break;
      }
   }
   return isConstType;
}

void adios2stream::Print(const Mesh& mesh, const mode print_mode)
{
   try
   {
      const bool isConstantType = IsConstantElementType(mesh.elements);

      if (!is_mesh_defined)
      {
         io.DefineAttribute<std::string>("info/format", "MFEM ADIOS2 BP");
         io.DefineAttribute<std::string>("info/version", "0.1");
         io.DefineAttribute<uint32_t>("dimension",
                                      static_cast<int32_t>(mesh.Dimension()) );

         // vertices
         io.DefineVariable<uint32_t>("NumOfVertices", {adios2::LocalValueDim});
         if (mesh.Nodes == NULL)
         {
            io.DefineVariable<double>(
            "mesh/vertices", {}, {},
            {static_cast<size_t>(mesh.NumOfVertices), static_cast<size_t>(mesh.spaceDim)});
         }

         // elements
         io.DefineVariable<uint32_t>("NumOfElements", {adios2::LocalValueDim});

         const size_t nElements = static_cast<size_t>(mesh.NumOfElements);
         const size_t nElementVertices =
            static_cast<size_t>(mesh.elements[0]->GetNVertices());

         if (isConstantType)
         {
            io.DefineVariable<uint64_t>("connectivity", {}, {},
            {nElements, nElementVertices+1});
            io.DefineVariable<uint32_t>("types");
         }
         else
         {
            throw std::invalid_argument("MFEM::adios2stream ERROR: non-constant element types not yet implemented\n");
         }
         is_mesh_defined = true;
      }

      if (!engine) // if Engine is closed
      {
         engine = io.Open(name, adios2::Mode::Write);
      }

      engine.Put("NumOfElements", static_cast<uint32_t>(mesh.NumOfElements));
      engine.Put("NumOfVertices", static_cast<uint32_t>(mesh.NumOfVertices));

      const uint32_t vtkType = GLVISToVTKType(static_cast<int>
                                              (mesh.elements[0]->GetGeometryType()));
      engine.Put("types", vtkType);

      if (mesh.Nodes == NULL)
      {
         adios2::Variable<double> varVertices =
            io.InquireVariable<double>("vertices");
         if (!varVertices)
         {
            varVertices = io.DefineVariable<double>("vertices", {}, {},
            {
               static_cast<size_t>(mesh.NumOfVertices),
               static_cast<size_t>(mesh.spaceDim)
            });
         }

         adios2::Variable<double>::Span spanVertices = engine.Put(varVertices);
         // vertices
         for (int v = 0; v < mesh.NumOfVertices; ++v)
         {
            for (int coord = 0; coord < mesh.spaceDim; ++coord)
            {
               spanVertices[v * mesh.spaceDim + coord] = mesh.vertices[v](coord);
            }
         }
      }
      else
      {
         mesh.Nodes->Save(*this, "vertices", data_type::none);
      }

      // use Span to save "vertices" and "connectivity"
      // from non-contiguous  "vertices" and "elements" arrays
      adios2::Variable<uint64_t> varConnectivity =
         io.InquireVariable<uint64_t>("connectivity");
      adios2::Variable<uint64_t>::Span spanConnectivity = engine.Put<uint64_t>
                                                          (varConnectivity);

      // connectivity
      size_t elementPosition = 0;
      for (int e = 0; e < mesh.NumOfElements; ++e)
      {
         const int nVertices = mesh.elements[e]->GetNVertices();
         spanConnectivity[elementPosition] = nVertices;
         for (int v = 0; v < nVertices; ++v)
         {
            spanConnectivity[elementPosition + v + 1] = mesh.elements[e]->GetVertices()[v];
         }
         elementPosition += nVertices + 1;
      }

      if (print_mode == mode::sync)
      {
         engine.PerformPuts();
      }
   }
   catch (std::exception& e)
   {
      const std::string warning =
         "MFEM: adios2stream exception caught, removing directory: " + name + "," +
         e.what();
      mfem_warning( warning.c_str());
   }
}

// PRIVATE
std::string adios2stream::VTKSchema() const noexcept
{
   std::string vtk_point_data_schema = vtk_schema_pre + "      <PointData>\n";
   for (const std::string& point_datum : point_data )
   {
      vtk_point_data_schema += "        <DataArray Name=\"" + point_datum +"\"/>\n";
   }
   vtk_point_data_schema += "      </PointData>\n" + vtk_schema_post;
   return vtk_point_data_schema;
}

}  // end namespace mfem
