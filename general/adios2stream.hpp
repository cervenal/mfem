// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.
//
// Created on: Jan 22, 2019
// Author: William F Godoy godoywf@ornl.gov
// adios2: Adaptable Input/Output System https://github.com/ornladios/ADIOS2

#ifndef MFEM_ADIOS2STREAM
#define MFEM_ADIOS2STREAM

#include "../config/config.hpp"

#include <map>
#include <memory>  // std::shared_ptr
#include <string>
#include <set>

#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif

#include <adios2.h>

namespace mfem
{

class Vector;
class FiniteElementSpace;
class GridFunction;
class Mesh;

template <class T>
class Array;
class Element;

#ifdef MFEM_USE_MPI
class ParGridFunction;
class ParMesh;
#endif

class adios2stream
{
   friend class Vector;
   friend class FiniteElementSpace;
   friend class GridFunction;
   friend class Mesh;

#ifdef MFEM_USE_MPI
   friend class ParGridFunction;
   friend class ParMesh;
#endif

public:
   /**
    * Open modes for adios2stream (from fstream)
    * out: write
    * in:  read
    * app: append
    */
   enum class openmode { out, in, app };

   /** Print and Save modes, deferred is done at Close or EndStep, sync is immediate */
   enum class mode {sync, deferred};

   enum class data_type {none, point_data};

#ifdef MFEM_USE_MPI

   /**
    * adios2stream MPI constructor, allows for passing parameters in source
    * code (compile-time) only.
    * @param name stream name
    * @param mode adios2stream::openmode::in (Read), adios2stream::openmode::out
    * (Write)
    * @param comm MPI communicator establishing domain for fstream
    * @param engineType available adios2 engine, default is BP3 file
    *        see https://adios2.readthedocs.io/en/latest/engines/engines.html
    * @throws std::invalid_argument (user input error) or std::runtime_error
    *         (system error)
    */
   adios2stream(const std::string& name, const openmode mode, MPI_Comm comm,
                const std::string engineType = "BPFile");
#else
   /**
    * adios2stream Non-MPI serial constructor, allows for passing parameters in
    * source code (compile-time) only.
    * @param name stream name
    * @param mode adios2stream::openmode::in (Read), adios2stream::openmode::out
    * (Write)
    * @param engineType available adios2 engine, default is BP3 file
    * @throws std::invalid_argument (user input error) or std::runtime_error
    *         (system error)
    */
   adios2stream(const std::string& name, const openmode mode,
                const std::string engineType = "BPFile");
#endif

   /** using RAII components, nothing to be deallocated **/
   virtual ~adios2stream();

   /**
    * Set parameters for a particular adios2stream Engine
    * See
    * https://adios2.readthedocs.io/en/latest/engines/engines.html#bp3-default
    * @param parameters map of key/value string elements
    */
   void SetParameters(const std::map<std::string, std::string>& parameters =
                         std::map<std::string, std::string>());

   /**
    * Single parameter version of SetParameters passing a key/value pair
    * See
    * https://adios2.readthedocs.io/en/latest/engines/engines.html#bp3-default
    * @param key input parameter key
    * @param value input parameter value
    */
   void SetParameter(const std::string key, const std::string value);

   /** Begins an I/O step */
   void BeginStep();

   /** Ends the current step, by default transports the data */
   void EndStep();

   /** Return the current step between BeginStep and EndStep */
   size_t CurrentStep() const noexcept;

   /** Finished interaction with adios2stream and flushes the data */
   void Close();

protected:

   std::set<std::string> point_data;

   int32_t GLVISToVTKType(const int glvisType) const noexcept;

   bool IsConstantElementType(const Array<Element*>& elements ) const noexcept;

   void Print(const Mesh& mesh, const adios2stream::mode print_mode = mode::sync);

private:
   /** placeholder for engine name */
   const std::string name;

   /** placeholder for engine openmode */
   const openmode adios2_openmode;

   /** main adios2 object that owns all the io and engine components */
   std::shared_ptr<adios2::ADIOS> adios;

   /** io object to set parameters, variables and engines */
   adios2::IO io;

   /** heavy object doing system-level I/O operations */
   adios2::Engine engine;

   /** true : engine step is active after engine.BeginStep(),
     *  false: inactive after engine.EndStep() */
   bool active_step = false;

   /** sets the current vtk_schema form point data arrays */
   std::string VTKSchema() const noexcept;

   bool is_mesh_defined = false;

   static const std::string vtk_schema_pre;

   static const std::string vtk_schema_post;

};

}  // end namespace mfem

#endif /* MFEM_ADIOS2STREAM */
