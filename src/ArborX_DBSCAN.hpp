/****************************************************************************
 * Copyright (c) 2017-2022 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_DBSCAN_HPP
#define ARBORX_DBSCAN_HPP

#include <ArborX_AccessTraits.hpp>
#include <ArborX_DetailsCartesianGrid.hpp>
#include <ArborX_DetailsFDBSCAN.hpp>
#include <ArborX_DetailsFDBSCANDenseBox.hpp>
#include <ArborX_DetailsHalfTraversal.hpp>
//#include <ArborX_DetailsHalfTraversalBruteForce.hpp>
#include <ArborX_DetailsSortUtils.hpp>
#include <ArborX_HyperBox.hpp>
#include <ArborX_HyperSphere.hpp>
#include <ArborX_LinearBVH.hpp>
#include <ArborX_BruteForce.hpp>
#include <ArborX_Sphere.hpp>
#include <variant>

namespace ArborX
{

namespace Details
{

// All points are marked as if they were core points minpts = 2 case.
// Obviously, this is not true. However, in the algorithms it is used only for
// pairs of points within the distance eps, in which case it is correct.
struct CCSCorePoints
{
  KOKKOS_FUNCTION bool operator()(int) const { return true; }
};

template <typename MemorySpace>
struct DBSCANCorePoints
{
  Kokkos::View<int *, MemorySpace> _num_neigh;
  int _core_min_size;

  KOKKOS_FUNCTION bool operator()(int const i) const
  {
    return _num_neigh(i) >= _core_min_size;
  }
};

template <typename Primitives>
struct PrimitivesWithRadius
{
  Primitives _primitives;
  float _r;
};

struct WithinRadiusGetter
{
  float _r;

  template <typename Box>
  KOKKOS_FUNCTION auto operator()(Box const &box) const
  {
    static_assert(GeometryTraits::is_box<Box>::value);

    constexpr int dim = GeometryTraits::dimension_v<Box>;
    auto const &hyper_point =
        reinterpret_cast<ExperimentalHyperGeometry::Point<dim> const &>(
            box.minCorner());
    using ArborX::intersects;
    return intersects(ExperimentalHyperGeometry::Sphere<dim>{hyper_point, _r});
  }
};

template <typename Primitives, typename PermuteFilter>
struct PrimitivesWithRadiusReorderedAndFiltered
{
  Primitives _primitives;
  float _r;
  PermuteFilter _filter;
};

// Mixed primitives consist of a set of boxes corresponding to dense cells,
// followed by boxes corresponding to points in non-dense cells.
template <typename PointPrimitives, typename DenseCellOffsets,
          typename CellIndices, typename Permutation>
struct MixedBoxPrimitives
{
  PointPrimitives _point_primitives;
  CartesianGrid<GeometryTraits::dimension_v<typename AccessTraitsHelper<
      AccessTraits<PointPrimitives, PrimitivesTag>>::type>>
      _grid;
  DenseCellOffsets _dense_cell_offsets;
  int _num_points_in_dense_cells; // to avoid lastElement() in AccessTraits
  CellIndices _sorted_cell_indices;
  Permutation _permute;
};

} // namespace Details

template <typename Primitives>
struct AccessTraits<Details::PrimitivesWithRadius<Primitives>, PredicatesTag>
{
  using PrimitivesAccess = AccessTraits<Primitives, PrimitivesTag>;

  using memory_space = typename PrimitivesAccess::memory_space;
  using Predicates = Details::PrimitivesWithRadius<Primitives>;

  static KOKKOS_FUNCTION size_t size(Predicates const &w)
  {
    return PrimitivesAccess::size(w._primitives);
  }
  static KOKKOS_FUNCTION auto get(Predicates const &w, size_t i)
  {
    auto const &point = PrimitivesAccess::get(w._primitives, i);
    constexpr int dim =
        GeometryTraits::dimension_v<std::decay_t<decltype(point)>>;
    // FIXME reinterpret_cast is dangerous here if access traits return user
    // point structure (e.g., struct MyPoint { float y; float x; })
    auto const &hyper_point =
        reinterpret_cast<ExperimentalHyperGeometry::Point<dim> const &>(point);
    return attach(
        intersects(ExperimentalHyperGeometry::Sphere<dim>{hyper_point, w._r}),
        (int)i);
  }
};

template <typename Primitives, typename PermuteFilter>
struct AccessTraits<Details::PrimitivesWithRadiusReorderedAndFiltered<
                        Primitives, PermuteFilter>,
                    PredicatesTag>
{
  using PrimitivesAccess = AccessTraits<Primitives, PrimitivesTag>;

  using memory_space = typename PrimitivesAccess::memory_space;
  using Predicates =
      Details::PrimitivesWithRadiusReorderedAndFiltered<Primitives,
                                                        PermuteFilter>;

  static KOKKOS_FUNCTION size_t size(Predicates const &w)
  {
    return w._filter.extent(0);
  }
  static KOKKOS_FUNCTION auto get(Predicates const &w, size_t i)
  {
    int index = w._filter(i);
    auto const &point = PrimitivesAccess::get(w._primitives, index);
    constexpr int dim =
        GeometryTraits::dimension_v<std::decay_t<decltype(point)>>;
    // FIXME reinterpret_cast is dangerous here if access traits return user
    // point structure (e.g., struct MyPoint { float y; float x; })
    auto const &hyper_point =
        reinterpret_cast<ExperimentalHyperGeometry::Point<dim> const &>(point);
    return attach(
        intersects(ExperimentalHyperGeometry::Sphere<dim>{hyper_point, w._r}),
        (int)index);
  }
};

template <typename PointPrimitives, typename MixedOffsets, typename CellIndices,
          typename Permutation>
struct AccessTraits<Details::MixedBoxPrimitives<PointPrimitives, MixedOffsets,
                                                CellIndices, Permutation>,
                    ArborX::PrimitivesTag>
{
  using Primitives = Details::MixedBoxPrimitives<PointPrimitives, MixedOffsets,
                                                 CellIndices, Permutation>;
  static KOKKOS_FUNCTION std::size_t size(Primitives const &w)
  {
    auto const &dco = w._dense_cell_offsets;

    auto const n = w._permute.size();
    auto num_dense_primitives = dco.size() - 1;
    auto num_sparse_primitives = n - w._num_points_in_dense_cells;

    return num_dense_primitives + num_sparse_primitives;
  }
  static KOKKOS_FUNCTION auto get(Primitives const &w, std::size_t i)
  {
    auto const &dco = w._dense_cell_offsets;

    auto num_dense_primitives = dco.size() - 1;
    if (i < num_dense_primitives)
    {
      // For a primitive corresponding to a dense cell, use that cell's box.
      // It may not be tight around the points inside, but is cheap to
      // compute.
      auto cell_index = w._sorted_cell_indices(dco(i));
      return w._grid.cellBox(cell_index);
    }

    // For a primitive corresponding to a point in a non-dense cell, use that
    // point. But first, figure out its index, which requires some
    // computations.
    using Access = AccessTraits<PointPrimitives, PrimitivesTag>;

    i = (i - num_dense_primitives) + w._num_points_in_dense_cells;

    auto const &point = Access::get(w._point_primitives, w._permute(i));
    constexpr int dim =
        GeometryTraits::dimension_v<std::decay_t<decltype(point)>>;
    // FIXME reinterpret_cast is dangerous here if access traits return user
    // point structure (e.g., struct MyPoint { float y; float x; })
    auto const &hyper_point =
        reinterpret_cast<ExperimentalHyperGeometry::Point<dim> const &>(point);
    return ExperimentalHyperGeometry::Box<dim>{hyper_point, hyper_point};
  }
  using memory_space = typename MixedOffsets::memory_space;
};

namespace DBSCAN
{

enum class Implementation
{
  FDBSCAN,
  FDBSCAN_DenseBox
};

struct Parameters
{
  // Print timers to standard output
  bool _verbose = false;
  // Algorithm implementation (FDBSCAN or FDBSCAN-DenseBox)
  Implementation _implementation = Implementation::FDBSCAN_DenseBox;
  // Tree type (construct-and-traverse bvh or use brute force)
  std::string _tree;

  Parameters &setVerbosity(bool verbose)
  {
    _verbose = verbose;
    return *this;
  }
  Parameters &setImplementation(Implementation impl)
  {
    _implementation = impl;
    return *this;
  }
  Parameters &setTree(std::string tree)
  {
    _tree = tree;
    return *this;
  }
};
} // namespace DBSCAN

// Return a variant which is either a brute or bvh object
template<typename MemorySpace, typename Box>
using brutebvh = std::variant<std::monostate , ArborX::BruteForce<MemorySpace, Box> , BasicBoundingVolumeHierarchy<MemorySpace, Box>>;

template <typename ExecutionSpace, typename Primitives >
brutebvh<typename ExecutionSpace::memory_space, 
	 ExperimentalHyperGeometry::Box<GeometryTraits::dimension_v< typename Details::AccessTraitsHelper< AccessTraits<Primitives, PrimitivesTag>>::type> >> 
         select_tree(const std::string tree, ExecutionSpace const &exec_space, Primitives const &primitives)
{

   using Access = AccessTraits<Primitives, PrimitivesTag>;
   using MemorySpace = typename Access::memory_space;
   static_assert(std::is_same_v<MemorySpace, typename ExecutionSpace::memory_space >);

   constexpr int dim = GeometryTraits::dimension_v<typename Details::AccessTraitsHelper<Access>::type>;
   using Box = ExperimentalHyperGeometry::Box<dim>;

   if (tree == "bvh") return BasicBoundingVolumeHierarchy<MemorySpace, Box>(exec_space, primitives);
   else if (tree == "brute") return ArborX::BruteForce<MemorySpace, Box>(exec_space, primitives);
   else return std::monostate();

   // Monostate should return error since this indicates an invalid choice
}


// Visitor for HalfTraversal
template <typename ExecutionSpace, typename Predicates, typename Callback, typename RadiusGetter, typename Brute, typename BVH, typename Primitives, typename CorePoints>
struct halftraversal_visitor
{

  ExecutionSpace const &exec_space;
  Predicates predicates;
  Callback callback;
  RadiusGetter radiusgetter;

  using Access = AccessTraits<Primitives, PrimitivesTag>;
  using MemorySpace = typename Access::memory_space;
//  using UnionFind = Details::UnionFind<MemorySpace>;

#ifdef KOKKOS_ENABLE_SERIAL
  using UnionFind = Details::UnionFind<MemorySpace,
      /*DoSerial=*/std::is_same_v<ExecutionSpace, Kokkos::Serial>>;
#else
  using UnionFind = Details::UnionFind<MemorySpace>;
#endif

//#if defined(KOKKOS_COMPILER_NVCC) && (KOKKOS_COMPILER_NVCC < 1140)
  using HalfTraversal = Details::HalfTraversal<
        BVH, Details::FDBSCANCallback<UnionFind, CorePoints>,
        Details::WithinRadiusGetter>;
//#else
//  using Details::HalfTraversal;
//#endif

  // If bru_or_bvh has a 'BVH' object, I want to call this:
  void operator()(BVH bvh){
     HalfTraversal(exec_space,
		   bvh,
		   callback,
		   radiusgetter);
  }
  // If bru_or_bvh has a 'Brute' object, I want to call this:
  void operator()(Brute brute){
// TO DO: fix issue with brute force callback
//     brute.query(exec_space, predicates, callback);
  }

  void operator()(std::monostate tree){};

};

// Visitor for query
template <typename ExecutionSpace, typename Predicates, typename Callback>
struct query_visitor
{

  ExecutionSpace const &exec_space;
  Predicates predicates;
  Callback callback;

  template<typename treetype>
  void operator()(treetype tree){
     tree.query(exec_space, predicates, callback);
  }
  void operator()(std::monostate tree){};

};
template <typename ExecutionSpace, typename Primitives>
Kokkos::View<int *,
             typename AccessTraits<Primitives, PrimitivesTag>::memory_space>
dbscan(ExecutionSpace const &exec_space, Primitives const &primitives,
       float eps, int core_min_size,
       DBSCAN::Parameters const &parameters = DBSCAN::Parameters())
{
  Kokkos::Profiling::pushRegion("ArborX::DBSCAN");

  using Access = AccessTraits<Primitives, PrimitivesTag>;
  using MemorySpace = typename Access::memory_space;

  static_assert(
      KokkosExt::is_accessible_from<MemorySpace, ExecutionSpace>::value,
      "Primitives must be accessible from the execution space");

  ARBORX_ASSERT(eps > 0);
  ARBORX_ASSERT(core_min_size >= 2);

#ifdef KOKKOS_ENABLE_SERIAL
  using UnionFind = Details::UnionFind<
      MemorySpace,
      /*DoSerial=*/std::is_same_v<ExecutionSpace, Kokkos::Serial>>;
#else
  using UnionFind = Details::UnionFind<MemorySpace>;
#endif

  constexpr int dim = GeometryTraits::dimension_v<
      typename Details::AccessTraitsHelper<Access>::type>;
  using Box = ExperimentalHyperGeometry::Box<dim>;

  bool const is_special_case = (core_min_size == 2);

  bool const verbose = parameters._verbose;

  int const n = Access::size(primitives);

  Kokkos::View<int *, MemorySpace> num_neigh("ArborX::DBSCAN::num_neighbors",
                                             0);

  Kokkos::View<int *, MemorySpace> labels(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "ArborX::DBSCAN::labels"),
      n);
  ArborX::iota(exec_space, labels);

  if (parameters._implementation == DBSCAN::Implementation::FDBSCAN)
  {
    // Build the tree
    Kokkos::Profiling::pushRegion("ArborX::DBSCAN::tree_construction");
//  PT: Variant decl.
    brutebvh<MemorySpace, Box> bru_or_bvh = select_tree(parameters._tree, exec_space, primitives);
//
//    BasicBoundingVolumeHierarchy<MemorySpace, Box> bvh(exec_space, primitives);
    Kokkos::Profiling::popRegion();

    // PT: predicates will go away in HalfTraversal version
    //auto const predicates = Details::PrimitivesWithRadius<Primitives>{primitives, eps};

    Kokkos::Profiling::pushRegion("ArborX::DBSCAN::clusters");
    if (is_special_case)
    {
      // Perform the queries and build clusters through callback
      using CorePoints = Details::CCSCorePoints;

/*
#if defined(KOKKOS_COMPILER_NVCC) && (KOKKOS_COMPILER_NVCC < 1140)
      // Workaround a compiler bug
      using HalfTraversal = Details::HalfTraversal<
          decltype(bvh), Details::FDBSCANCallback<UnionFind, CorePoints>,
          Details::WithinRadiusGetter>;
#else
      using Details::HalfTraversal;
#endif
*/
      Kokkos::Profiling::pushRegion("ArborX::DBSCAN::clusters::query");
      // Query via call to visitor (worked previously)
      // Templated above via: template <typename ExecutionSpace, typename Predicates, typename Callback>
//      std::visit( query_visitor
//                        <
//                         ExecutionSpace,
//                         Details::PrimitivesWithRadius<Primitives>,
//                         Details::FDBSCANCallback<UnionFind, CorePoints>
//                        >
//                        {
//                         exec_space,
//                         predicates,
//                         Details::FDBSCANCallback<UnionFind, CorePoints>{labels, core_points}
//                        }, bru_or_bvh );

      // PT halftraversal_visitor UNDER CONSTRUCTION
      CorePoints core_points; // tmp
      std::visit( halftraversal_visitor
                        <
                         ExecutionSpace,                                     // typename for ExecutionSpace  
                         Details::PrimitivesWithRadius<Primitives>,          // typename for Predicates
                         Details::FDBSCANCallback<UnionFind, CorePoints>,    // typename for Callback
                         Details::WithinRadiusGetter,                        // typename for RadiusGetter
                         ArborX::BruteForce<MemorySpace, Box>,               // typename for Brute -- define in visitor?
                         BasicBoundingVolumeHierarchy<MemorySpace, Box>,     // typename for BVH   -- define in visitor?
			 Primitives,                                         // typename for Primitives
			 Details::CCSCorePoints                              // typename for CorePoints
                        >
                        {
                         exec_space,                                         // instance of ExecutionSpace
                         Details::PrimitivesWithRadius<Primitives>{primitives, eps}, // instance of Predicates
                         Details::FDBSCANCallback<UnionFind, CorePoints>{labels, core_points}, // instance of Callback
                         Details::WithinRadiusGetter{eps}                    // instance of RadiusGetter
                        }, bru_or_bvh );

   // PT: replace above with HalfTraversal visitor 
//      HalfTraversal(
//          exec_space, bvh,
//          Details::FDBSCANCallback<UnionFind, CorePoints>{labels, CorePoints{}},
//          Details::WithinRadiusGetter{eps});
//      Kokkos::Profiling::popRegion();
    }
    else
    {
      auto const predicates =
          Details::PrimitivesWithRadius<Primitives>{primitives, eps};

      // Determine core points
      Kokkos::Profiling::pushRegion("ArborX::DBSCAN::clusters::num_neigh");
      Kokkos::resize(Kokkos::view_alloc(exec_space), num_neigh, n);
      // Query via call to visitor
      std::visit( query_visitor<ExecutionSpace, Details::PrimitivesWithRadius<Primitives>,
                  Details::CountUpToN<MemorySpace>>{exec_space, predicates,
                  Details::CountUpToN<MemorySpace>{num_neigh, core_min_size}},
                  bru_or_bvh );
      // PT: visitor will eventually replace query call below
//      bvh.query(exec_space, predicates,
//                Details::CountUpToN<MemorySpace>{num_neigh, core_min_size});
      Kokkos::Profiling::popRegion();

      using CorePoints = Details::DBSCANCorePoints<MemorySpace>;
/*
#if defined(KOKKOS_COMPILER_NVCC) && (KOKKOS_COMPILER_NVCC < 1140)
      // Workaround a compiler bug
      using HalfTraversal = Details::HalfTraversal<
          decltype(bvh), Details::FDBSCANCallback<UnionFind, CorePoints>,
          Details::WithinRadiusGetter>;
#else
      using Details::HalfTraversal;
#endif
*/
      // Perform the queries and build clusters through callback
      Kokkos::Profiling::pushRegion("ArborX::DBSCAN::clusters::query");
      // Query via call to visitor
//      std::visit( query_visitor<ExecutionSpace, Details::PrimitivesWithRadius<Primitives>,
//                  Details::FDBSCANCallback<UnionFind, CorePoints>>{exec_space, predicates,
//                  Details::FDBSCANCallback<UnionFind, CorePoints>{labels, CorePoints{num_neigh, core_min_size}}},
//                  bru_or_bvh );
      // PT halftraversal_visitor UNDER CONSTRUCTION
      std::visit( halftraversal_visitor
                        <
                         ExecutionSpace,                                     // typename for ExecutionSpace  
                         Details::PrimitivesWithRadius<Primitives>,          // typename for Predicates
                         Details::FDBSCANCallback<UnionFind, CorePoints>,    // typename for Callback
                         Details::WithinRadiusGetter,                        // typename for RadiusGetter
                         ArborX::BruteForce<MemorySpace, Box>,               // typename for Brute -- define in visitor?
                         BasicBoundingVolumeHierarchy<MemorySpace, Box>,     // typename for BVH   -- define in visitor?
                         Primitives,                                         // typename for Primitives
			 Details::DBSCANCorePoints<MemorySpace>              // typename for CorePoints
                        >
                        {
                         exec_space,                                         // instance of ExecutionSpace
                         Details::PrimitivesWithRadius<Primitives>{primitives, eps}, // instance of Predicates
                         Details::FDBSCANCallback<UnionFind, CorePoints>{labels, CorePoints{num_neigh, core_min_size}}, // instance of Callback
                         Details::WithinRadiusGetter{eps}                    // instance of RadiusGetter
                        }, bru_or_bvh );

      // PT: replace above with HalfTraversal visitor
//      HalfTraversal(exec_space, bvh,
//                    Details::FDBSCANCallback<UnionFind, CorePoints>{
//                        labels, CorePoints{num_neigh, core_min_size}},
//                    Details::WithinRadiusGetter{eps});
//      Kokkos::Profiling::popRegion();
    }
  }
  else if (parameters._implementation ==
           DBSCAN::Implementation::FDBSCAN_DenseBox)
  {
    // Find dense boxes
    Kokkos::Profiling::pushRegion("ArborX::DBSCAN::dense_cells");
    Box bounds;
    Details::TreeConstruction::calculateBoundingBoxOfTheScene(
        exec_space, Details::Indexables<Primitives>{primitives}, bounds);

    // The cell length is chosen to be eps/sqrt(dimension), so that any two
    // points within the same cell are within eps distance of each other.
    float const h = eps / std::sqrt(dim);
    Details::CartesianGrid<dim> const grid(bounds, h);

    auto cell_indices =
        Details::computeCellIndices(exec_space, primitives, grid);

    auto permute = Details::sortObjects(exec_space, cell_indices);
    auto &sorted_cell_indices = cell_indices; // alias

    int num_nonempty_cells;
    int num_points_in_dense_cells;
    {
      // Reorder indices and permutation so that the dense cells go first
      Kokkos::View<int *, MemorySpace> cell_offsets(
          "ArborX::DBSCAN::cell_offsets", 0);
      Details::computeOffsetsInOrderedView(exec_space, sorted_cell_indices,
                                           cell_offsets);
      num_nonempty_cells = cell_offsets.size() - 1;

      num_points_in_dense_cells = Details::reorderDenseAndSparseCells(
          exec_space, cell_offsets, core_min_size, sorted_cell_indices,
          permute);
    }
    int num_points_in_sparse_cells = n - num_points_in_dense_cells;

    auto dense_sorted_cell_indices = Kokkos::subview(
        sorted_cell_indices, Kokkos::make_pair(0, num_points_in_dense_cells));

    Kokkos::View<int *, MemorySpace> dense_cell_offsets(
        "ArborX::DBSCAN::dense_cell_offsets", 0);
    Details::computeOffsetsInOrderedView(exec_space, dense_sorted_cell_indices,
                                         dense_cell_offsets);
    int num_dense_cells = dense_cell_offsets.size() - 1;
    if (verbose)
    {
      printf("h = %e, n = [%zu", h, grid.extent(0));
      for (int d = 1; d < decltype(grid)::dim; ++d)
        printf(", %zu", grid.extent(d));
      printf("]\n");
      printf("#nonempty cells     : %10d\n", num_nonempty_cells);
      printf("#dense cells        : %10d [%.2f%%]\n", num_dense_cells,
             (100.f * num_dense_cells) / num_nonempty_cells);
      printf("#dense cell points  : %10d [%.2f%%]\n", num_points_in_dense_cells,
             (100.f * num_points_in_dense_cells) / n);
      printf("#mixed primitives   : %10d\n",
             num_dense_cells + num_points_in_sparse_cells);
    }

    Details::unionFindWithinEachDenseCell(exec_space, dense_sorted_cell_indices,
                                          permute, UnionFind{labels});

    Kokkos::Profiling::popRegion();

    // Build the tree
    Kokkos::Profiling::pushRegion("ArborX::DBSCAN::tree_construction");
    BasicBoundingVolumeHierarchy<MemorySpace, Box> bvh(
        exec_space,
        Details::MixedBoxPrimitives<Primitives, decltype(dense_cell_offsets),
                                    decltype(cell_indices), decltype(permute)>{
            primitives, grid, dense_cell_offsets, num_points_in_dense_cells,
            sorted_cell_indices, permute});

    Kokkos::Profiling::popRegion();

    Kokkos::Profiling::pushRegion("ArborX::DBSCAN::clusters");

    if (is_special_case)
    {
      // Perform the queries and build clusters through callback
      using CorePoints = Details::CCSCorePoints;
      Kokkos::Profiling::pushRegion("ArborX::DBSCAN::clusters::query");
      auto const predicates =
          Details::PrimitivesWithRadius<Primitives>{primitives, eps};
      bvh.query(
          exec_space, predicates,
          Details::FDBSCANDenseBoxCallback<UnionFind, CorePoints, Primitives,
                                           decltype(dense_cell_offsets),
                                           decltype(permute)>{
              labels, CorePoints{}, primitives, dense_cell_offsets, exec_space,
              permute, eps});
      Kokkos::Profiling::popRegion();
    }
    else
    {
      // Determine core points
      Kokkos::Profiling::pushRegion("ArborX::DBSCAN::clusters::num_neigh");
      Kokkos::resize(Kokkos::view_alloc(exec_space), num_neigh, n);
      // Set num neighbors for points in dense cells to max, so that they are
      // automatically core points
      Kokkos::parallel_for(
          "ArborX::DBSCAN::mark_dense_cells_core_points",
          Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0,
                                              num_points_in_dense_cells),
          KOKKOS_LAMBDA(int i) { num_neigh(permute(i)) = INT_MAX; });
      // Count neighbors for points in sparse cells
      auto sparse_permute = Kokkos::subview(
          permute, Kokkos::make_pair(num_points_in_dense_cells, n));

      auto const sparse_predicates =
          Details::PrimitivesWithRadiusReorderedAndFiltered<
              Primitives, decltype(sparse_permute)>{primitives, eps,
                                                    sparse_permute};
      bvh.query(exec_space, sparse_predicates,
                Details::CountUpToN_DenseBox<MemorySpace, Primitives,
                                             decltype(dense_cell_offsets),
                                             decltype(permute)>(
                    num_neigh, primitives, dense_cell_offsets, permute,
                    core_min_size, eps, core_min_size));
      Kokkos::Profiling::popRegion();

      using CorePoints = Details::DBSCANCorePoints<MemorySpace>;

      // Perform the queries and build clusters through callback
      Kokkos::Profiling::pushRegion("ArborX::DBSCAN::clusters::query");
      auto const predicates =
          Details::PrimitivesWithRadius<Primitives>{primitives, eps};
      bvh.query(
          exec_space, predicates,
          Details::FDBSCANDenseBoxCallback<UnionFind, CorePoints, Primitives,
                                           decltype(dense_cell_offsets),
                                           decltype(permute)>{
              labels, CorePoints{num_neigh, core_min_size}, primitives,
              dense_cell_offsets, exec_space, permute, eps});
      Kokkos::Profiling::popRegion();
    }
  }

  // Per [1]:
  //
  // ```
  // The finalization kernel will, ultimately, make all parents
  // point directly to the representative.
  // ```
  Kokkos::View<int *, MemorySpace> cluster_sizes(
      Kokkos::view_alloc(exec_space, "ArborX::DBSCAN::cluster_sizes"), n);
  Kokkos::parallel_for(
      "ArborX::DBSCAN::finalize_labels",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
      KOKKOS_LAMBDA(int const i) {
        // ##### ECL license (see LICENSE.ECL) #####
        int next;
        int vstat = labels(i);
        int const old = vstat;
        while (vstat > (next = labels(vstat)))
        {
          vstat = next;
        }
        if (vstat != old)
          labels(i) = vstat;

        Kokkos::atomic_increment(&cluster_sizes(labels(i)));
      });
  if (is_special_case)
  {
    // Ideally, this kernel would have had the exactly same form as in the
    // else() clause. But there's no available valid is_core() for use here:
    // - CCSCorePoints cannot be used as it always returns true, which is OK
    //   inside the callback, but not here
    // - DBSCANCorePoints cannot be used either as num_neigh is not initialized
    //   in the special case.
    Kokkos::parallel_for(
        "ArborX::DBSCAN::mark_noise",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
        KOKKOS_LAMBDA(int const i) {
          if (cluster_sizes(labels(i)) == 1)
            labels(i) = -1;
        });
  }
  else
  {
    Details::DBSCANCorePoints<MemorySpace> is_core{num_neigh, core_min_size};
    Kokkos::parallel_for(
        "ArborX::DBSCAN::mark_noise",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
        KOKKOS_LAMBDA(int const i) {
          if (cluster_sizes(labels(i)) == 1 && !is_core(i))
            labels(i) = -1;
        });
  }
  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::popRegion();

  return labels;
}

} // namespace ArborX

#endif
