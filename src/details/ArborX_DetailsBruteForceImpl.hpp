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

  template <class ExecutionSpace, class Primitives, class Predicates,
            class Callback>
  static void queryN(ExecutionSpace const &space, Primitives const &primitives,
                    Predicates const &predicates, Callback const &callback)
  {
    using TeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
    using AccessPrimitives = AccessTraits<Primitives, PrimitivesTag>;
    using AccessPredicates = AccessTraits<Predicates, PredicatesTag>;
    using PredicateType = typename AccessTraitsHelper<AccessPredicates>::type;
    using PrimitiveType = typename AccessTraitsHelper<AccessPrimitives>::type;

    using MemorySpacePrimitives = typename AccessPrimitives::memory_space;
    using MemorySpacePredicates = typename AccessPredicates::memory_space;

    int const n_primitives = AccessPrimitives::size(primitives);
    int const n_predicates = AccessPredicates::size(predicates);

    const char tA[] = {"N"};
    const char tB[] = {"T"};

    constexpr int DIM = GeometryTraits::dimension_v<
            typename Details::AccessTraitsHelper<AccessPrimitives>::type>;

    const double alpha = 1.0;
    const double beta = 0.0;

    Kokkos::View<PrimitiveType **, Kokkos::LayoutLeft, MemorySpacePrimitives> A ("primitives view", n_primitives, DIM );
    Kokkos::View<PredicateType **, Kokkos::LayoutLeft, MemorySpacePredicates> B ("predicates view", n_predicates, DIM );
// PT: check type, add memoryspace to C
    Kokkos::View<float **, Kokkos::LayoutLeft> C ("distance view", n_primitives, n_predicates );

// Fill A and B arrays, with primitives and predicates, respectively.
    Kokkos::parallel_for("ArborX::BruteForce::query::spatial::"
                         "fill_primitives_array",
                         Kokkos::RangePolicy<ExecutionSpace>(
                         space, 0, n_primitives),
                         KOKKOS_LAMBDA(int q){

        // Primitive for this thread
        auto const &primitive = AccessPrimitives::get(primitives, q);
        for (int j=0; j<DIM; j++){
// PT: build issue below: new brute/bvh interface needed to fix
            A(q,j) = primitive[j];
	}
    });

/*
    Kokkos::parallel_for("ArborX::BruteForce::query::spatial::"
                         "fill_predicates_array",
                         Kokkos::RangePolicy<ExecutionSpace>(
                         space, 0, n_predicates*DIM),
                         KOKKOS_LAMBDA(int q){

        // Predicate and dimension for this thread (compare performance)
	int i =  q / DIM;
        int j =  q % DIM;
        auto const &predicate = AccessPredicates::get(predicates, i);
	B(i,j) = predicate(j);
    });

// Call gemm to compute the distances
//    KokkosBlas::gemm(space, tA, tB, alpha, A, B, beta, C );
*/

// PT: callback here
// C is the matrix of distances between primitives in A and predicates in B.

// PT add destructors for A,B,C

  }

};
} // namespace Details
} // namespace ArborX

#endif
