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

#ifndef ARBORX_DETAILS_BRUTE_FORCE_IMPL_HPP
#define ARBORX_DETAILS_BRUTE_FORCE_IMPL_HPP

#include <ArborX_AccessTraits.hpp>
#include <ArborX_DetailsAlgorithms.hpp> // expand
#include <ArborX_Exception.hpp>
#include <Kokkos_Core.hpp>
#include <KokkosBlas3_gemm.hpp>

namespace ArborX
{
namespace Details
{
struct BruteForceImpl
{
  template <class ExecutionSpace, class Values, class IndexableGetter,
            class Nodes, class Bounds>
  static void initializeBoundingVolumesAndReduceBoundsOfTheScene(
      ExecutionSpace const &space, Values const &values,
      IndexableGetter const &indexable_getter, Nodes const &nodes,
      Bounds &bounds)
  {
    Kokkos::parallel_reduce(
        "ArborX::BruteForce::BruteForce::"
        "initialize_values_and_reduce_bounds",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, values.size()),
        KOKKOS_LAMBDA(int i, Bounds &update) {
          nodes(i) = values(i);

          using Details::expand;
          Bounds bounding_volume{};
          expand(bounding_volume, indexable_getter(nodes(i)));
          update += bounding_volume;
        },
        Kokkos::Sum<Bounds>{bounds});
  }

  template <class ExecutionSpace, class Predicates, class Values,
            class Indexables, class Callback>
  static void query_shared(ExecutionSpace const &space, Predicates const &predicates,
                    Values const &values, Indexables const &indexables,
                    Callback const &callback)
  {
    using TeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
    using AccessPredicates = AccessTraits<Predicates, PredicatesTag>;
    using PredicateType = typename AccessTraitsHelper<AccessPredicates>::type;
    using IndexableType = std::decay_t<decltype(indexables(0))>;

    int const n_indexables = values.size();
    int const n_predicates = AccessPredicates::size(predicates);
    int max_scratch_size = TeamPolicy::scratch_size_max(0);
    // half of the scratch memory used by predicates and half for indexables
    int const predicates_per_team =
        max_scratch_size / 2 / sizeof(PredicateType);
    int const indexables_per_team =
        max_scratch_size / 2 / sizeof(IndexableType);
    ARBORX_ASSERT(predicates_per_team > 0);
    ARBORX_ASSERT(indexables_per_team > 0);

    int const n_indexable_tiles =
        std::ceil((float)n_indexables / indexables_per_team);
    int const n_predicate_tiles =
        std::ceil((float)n_predicates / predicates_per_team);
    int const n_teams = n_indexable_tiles * n_predicate_tiles;

    using ScratchPredicateType =
        Kokkos::View<PredicateType *,
                     typename ExecutionSpace::scratch_memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using ScratchIndexableType =
        Kokkos::View<IndexableType *,
                     typename ExecutionSpace::scratch_memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    int scratch_size = ScratchPredicateType::shmem_size(predicates_per_team) +
                       ScratchIndexableType::shmem_size(indexables_per_team);

    Kokkos::parallel_for(
        "ArborX::BruteForce::query::spatial::"
        "check_all_predicates_against_all_indexables",
        TeamPolicy(space, n_teams, Kokkos::AUTO, 1)
            .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(typename TeamPolicy::member_type const &teamMember) {
          // select the tiles of predicates/indexables checked by each team
          int predicate_start = predicates_per_team *
                                (teamMember.league_rank() / n_indexable_tiles);
          int indexable_start = indexables_per_team *
                                (teamMember.league_rank() % n_indexable_tiles);

          int predicates_in_this_team = KokkosExt::min(
              predicates_per_team, n_predicates - predicate_start);
          int indexables_in_this_team = KokkosExt::min(
              indexables_per_team, n_indexables - indexable_start);

          ScratchPredicateType scratch_predicates(teamMember.team_scratch(0),
                                                  predicates_per_team);
          ScratchIndexableType scratch_indexables(teamMember.team_scratch(0),
                                                  indexables_per_team);
          // fill the scratch space with the predicates / indexables in the tile
          Kokkos::parallel_for(
              Kokkos::TeamVectorRange(teamMember, predicates_in_this_team),
              [&](const int q) {
                scratch_predicates(q) =
                    AccessPredicates::get(predicates, predicate_start + q);
              });
          Kokkos::parallel_for(
              Kokkos::TeamVectorRange(teamMember, indexables_in_this_team),
              [&](const int j) {
                scratch_indexables(j) = indexables(indexable_start + j);
              });
          teamMember.team_barrier();

          // start threads for every predicate / indexable combination
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(teamMember, indexables_in_this_team),
              [&](int j) {
                Kokkos::parallel_for(
                    Kokkos::ThreadVectorRange(teamMember,
                                              predicates_in_this_team),
                    [&](const int q) {
                      auto const &predicate = scratch_predicates(q);
                      auto const &indexable = scratch_indexables(j);
                      if (predicate(indexable))
                      {
                        callback(predicate, values(indexable_start + j));
                      }
                    });
              });
        });
  }

  template <class ExecutionSpace, class Predicates, class Values,
            class Indexables, class Callback>
  static void query_tc_expt(ExecutionSpace const &space, Predicates const &predicates,
                    Values const &values, Indexables const &indexables,
                    Callback const &callback)
  {

    using AccessIndexables = AccessValues<Values>;
    using AccessPredicates = AccessTraits<Predicates, PredicatesTag>;
    using IndexableType = std::decay_t<decltype(indexables(0))>;
    using IndexableValueType = typename GeometryTraits::coordinate_type<IndexableType>::type;
    using PredicateValueType = std::decay_t<decltype(ArborX::getGeometry(AccessPredicates::get(predicates, 0))._centroid[0])>;
    using MemorySpaceIndexables = typename AccessIndexables::memory_space;
    using MemorySpacePredicates = typename AccessPredicates::memory_space;

    int const n_indexables = values.size();
    int const n_predicates = AccessPredicates::size(predicates);
    constexpr int DIM = GeometryTraits::dimension_v<IndexableType>;
    const char tA[] = {"N"};
    const char tB[] = {"T"};
    const IndexableValueType alpha = -2.0;
    const IndexableValueType beta = 1.0;

    Kokkos::View<IndexableValueType **, Kokkos::LayoutLeft, MemorySpaceIndexables> Asquared ("indexables squared view", n_indexables, 1);
    Kokkos::View<PredicateValueType **, Kokkos::LayoutLeft, MemorySpacePredicates> Bsquared ("predicates squared view", n_predicates, 1);
    Kokkos::View<IndexableValueType **, Kokkos::LayoutLeft, MemorySpaceIndexables> A ("indexables view", n_indexables, DIM );
    Kokkos::View<PredicateValueType **, Kokkos::LayoutLeft, MemorySpacePredicates> B ("predicates view", n_predicates, DIM );
    Kokkos::View<IndexableValueType **, Kokkos::LayoutLeft, MemorySpaceIndexables> C ("intersection view", n_indexables, n_predicates );

// Fill array containing sums-of-squares of indexables
    Kokkos::parallel_for("ArborX::BruteForce::query::spatial::"
                         "fill_indexables_sums_of_squares_array",
                         Kokkos::RangePolicy<ExecutionSpace>(
                         space, 0, n_indexables),
                         KOKKOS_LAMBDA(int i){

        // Indexable for this thread
        auto const &indexable = indexables(i);
	Asquared(i,1) = 0.0;

        for (int j=0; j<DIM; j++){
            Asquared(i,1) += indexable[j]*indexable[j];
	    }
    });

// Fill array 'Bsquared' with ones
    Kokkos::parallel_for("ArborX::BruteForce::query::spatial::"
                         "fill_ones_array_npredicates_size",
                         Kokkos::RangePolicy<ExecutionSpace>(
                         space, 0, n_predicates),
                         KOKKOS_LAMBDA(int i){

        Bsquared(i,1) = 1.0;
    });


// gemm to initialize results array with Asquared values
   KokkosBlas::gemm(space, tA, tB, 1.0, Asquared, Bsquared, 0.0, C );

// Fill array containing -r-squared + sums-of-squares of predicates
    Kokkos::parallel_for("ArborX::BruteForce::query::spatial::"
                         "fill_predicates_sums_of_squares_array",
                         Kokkos::RangePolicy<ExecutionSpace>(
                         space, 0, n_predicates),
                         KOKKOS_LAMBDA(int i){

        // Predicate for this thread
        auto const &predicate = AccessPredicates::get(predicates, i);
        auto const &geometry = ArborX::getGeometry(predicate);
        Bsquared(i,1) = -geometry._radius*geometry._radius;

        for (int j=0; j<DIM; j++){
            Bsquared(i,1) += geometry._centroid[j]*geometry._centroid[j];
            }
    });

// Fill array 'Asquared' with ones
    Kokkos::parallel_for("ArborX::BruteForce::query::spatial::"
                         "fill_ones_array_nindexables_size",
                         Kokkos::RangePolicy<ExecutionSpace>(
                         space, 0, n_indexables),
                         KOKKOS_LAMBDA(int i){

        Asquared(i,1) = 1.0;
    });

// gemm to accumulate results array with Bsquared values
   KokkosBlas::gemm(space, tA, tB, 1.0, Asquared, Bsquared, 1.0, C );

// Fill A and B arrays, with indexables and predicates, respectively.
    Kokkos::parallel_for("ArborX::BruteForce::query::spatial::"
                         "fill_indexables_array",
                         Kokkos::RangePolicy<ExecutionSpace>(
                         space, 0, n_indexables*DIM),
                         KOKKOS_LAMBDA(int q){

        // Indexable and dimension for this thread
        int i =  q / DIM;
        int j =  q % DIM;
        auto const &indexable = indexables(i);
        A(i,j) = indexable[j];
    });

    Kokkos::parallel_for("ArborX::BruteForce::query::spatial::"
                         "fill_predicates_array",
                         Kokkos::RangePolicy<ExecutionSpace>(
                         space, 0, n_predicates*DIM),
                         KOKKOS_LAMBDA(int q){

        // Predicate and dimension for this thread
	int i =  q / DIM;
        int j =  q % DIM;
        auto const &predicate = AccessPredicates::get(predicates, i);
        auto const &geometry = ArborX::getGeometry(predicate);
	B(i,j) = geometry._centroid[j]; 
    });

// Call gemm to compute the distances
    KokkosBlas::gemm(space, tA, tB, alpha, A, B, 1.0, C );

// Compare distances and callback
    Kokkos::parallel_for("ArborX::BruteForce::query::spatial::"
                         "check_intersections_and_callback",
                         Kokkos::RangePolicy<ExecutionSpace>(
                         space, 0, n_indexables*n_predicates),
                         KOKKOS_LAMBDA(int q){

        // Indexable and predicate for this thread
        int i =  q / n_predicates;
        int j =  q % n_predicates;
        if (C(i,j) <= 0)
        {
           auto const &predicate = AccessPredicates::get(predicates, j);
           callback(predicate, values(i));
        }
    });

  }

  template <class ExecutionSpace, class Predicates, class Values,
            class Indexables, class Callback>
  static void query(ExecutionSpace const &space, Predicates const &predicates,
                    Values const &values, Indexables const &indexables,
                    Callback const &callback)
  {
    using TeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
    using AccessPredicates = AccessTraits<Predicates, PredicatesTag>;
    using PredicateType = typename AccessTraitsHelper<AccessPredicates>::type;
    using IndexableType = std::decay_t<decltype(indexables(0))>;

    int const n_indexables = values.size();
    int const n_predicates = AccessPredicates::size(predicates);
    constexpr int DIM = GeometryTraits::dimension_v<IndexableType>;
    int max_scratch_size = TeamPolicy::scratch_size_max(0);

    // Hard-code predicates-per-team; adjust indexables-per-team accordingly
    // A100-specific tuning from HW/SDK limits/recommendations:
    // 2048 threads-per-SM / 128 predicates_per_team = 16 teams-per-SM (with 1 predicate-per-team)
    // 164 KB-per-SM / 16 teams-per-SM = 10.25 KB-per-team = 10496 bytes
    // 10496 bytes - 16 bytes-static-shared-memory - 1044 bytes-driver-shared-memory = 9436 bytes-per-team
    int const predicates_per_team = 128;
    int const indexables_per_team = 32 * std::floor((float(9436) / float(sizeof(IndexableType)) / float(32)));
//    int const indexables_per_team = 320;

    int const n_indexable_tiles =
        std::ceil((float)n_indexables / indexables_per_team);
    int const n_predicate_tiles =
        std::ceil((float)n_predicates / predicates_per_team);
    int const n_teams = n_predicate_tiles;

    // Just request scratch for indexables only
    using ScratchIndexableType =
        Kokkos::View<IndexableType *,
                     typename ExecutionSpace::scratch_memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    int scratch_size = ScratchIndexableType::shmem_size(indexables_per_team);

    Kokkos::parallel_for(
        "ArborX::BruteForce::query::spatial::"
        "all-predicates-vs-all-indexables-shared-128-by-indexable-tile",
        TeamPolicy(space, n_teams, 128 /*Kokkos::AUTO*/, 1)
            .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(typename TeamPolicy::member_type const &teamMember) {

          // select the tiles of predicates checked by each team
          int predicate_start = predicates_per_team * teamMember.league_rank();
          int predicates_in_this_team = KokkosExt::min(
              predicates_per_team, n_predicates - predicate_start);

          // just get the predicate from global, but load indexables from shared
          ScratchIndexableType scratch_indexables(teamMember.team_scratch(0),
                                                  indexables_per_team);

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(teamMember, predicates_in_this_team),
              [&](int q) {

                  // Predicate for this thread
                  auto const &predicate = AccessPredicates::get(predicates, predicate_start + q);

                  // Loop over primitive tiles
                  for (int i=0; i<n_indexables; i+=indexables_per_team){

                      int indexable_start = i;
                      int indexables_in_this_team = KokkosExt::min(
                      indexables_per_team, n_indexables - indexable_start);

                      // Load a new tile of primitives into scratch
                      Kokkos::parallel_for(
                          Kokkos::TeamVectorRange(teamMember, indexables_in_this_team),
                          [&](const int j) {
                            scratch_indexables(j) = indexables(indexable_start + j);
                         });
                      teamMember.team_barrier();

                      Kokkos::parallel_for(
                          Kokkos::ThreadVectorRange(teamMember, indexables_in_this_team),
                          [&](const int j) {
                            auto const &indexable = scratch_indexables(j);
                            if (predicate(indexable))
                            {
                              callback(predicate, values(indexable_start + j));
                            }
                          });
                      teamMember.team_barrier();
                  }
              });
        });
  }

  template <class ExecutionSpace, class Predicates, class Values,
            class Indexables, class Callback>
  static void query_noshared(ExecutionSpace const &space, Predicates const &predicates,
                    Values const &values, Indexables const &indexables,
                    Callback const &callback)
  {

    using AccessPredicates = AccessTraits<Predicates, PredicatesTag>;

    int const n_indexables = values.size();
    int const n_predicates = AccessPredicates::size(predicates);

    // This version does not use shared memory 
    // Each predicate checked against all primitives, with 1-work-item-per-predicate
    Kokkos::parallel_for("ArborX::BruteForce::query::spatial::"
		         "check_all_predicates_against_all_primitives_noshared",
                         Kokkos::RangePolicy<ExecutionSpace>(
                             space, 0, n_predicates),
                         KOKKOS_LAMBDA(int q){

        // Predicate for this thread
        auto const &predicate = AccessPredicates::get(predicates, q);

	// Check against primitives
	for (int i=0; i<n_indexables; i++){
	    auto const &primitive = indexables(i);
            if (predicate(primitive)){
               callback(predicate, values(i));
            }
	}
    });
  }

};
} // namespace Details
} // namespace ArborX

#endif
