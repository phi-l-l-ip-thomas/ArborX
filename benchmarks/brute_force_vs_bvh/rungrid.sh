#!/bin/bash

# Paths to Kokkos profiling tools (if requested)
NVCONNECTOR=/global/homes/p/pthomas/myktools/kokkos-tools/profiling/nvprof-connector/kp_nvprof_connector.so
STCONNECTOR=/global/homes/p/pthomas/myktools/kokkos-tools/profiling/space-time-stack/kp_space_time_stack.so

# Directory where to place tests (should not have underscore '_' in name)
TESTDIR='gridtestkp'

# Profiler options:
# 'nvidia': use NSight Systems to profile dbscan; extract timings from Systems output
# 'kp'    : extract timings from kp_space_time_stack output
# 'none'  : extract timings from verbose dbscan output
PROFILER='kp'

# Create test directory
if [ ! -d "$TESTDIR" ]; then
  mkdir ${TESTDIR}
fi

# Profiler specific options
if [ ${PROFILER,,} == 'nvidia' ]; then
export KOKKOS_PROFILE_LIBRARY=${NVCONNECTOR}
elif [ ${PROFILER,,} == 'kp' ]; then
export KOKKOS_PROFILE_LIBRARY=${STCONNECTOR}
elif [ ${PROFILER,,} == 'none' ]; then
export KOKKOS_PROFILE_LIBRARY=
else
echo 'unrecognized profiler option ${PROFILER}'
exit 1
fi

for TREE in brute bvh
do

for POINTS in 0000128 0000256 0000512 0001024 0002048 0004096 0008192 0016384 0032768 0065536 0131072 0262144 0524288 1048576
do

for EPS in 0.001 0.002 0.003 0.004 0.006 0.008 0.012 0.016 0.024 0.032 0.048 0.064 0.096 0.128 0.196 0.256 0.384 0.512 0.768 1.024 1.536 1.733
do

echo ${TREE}_${EPS}_${POINTS}
if [ ${PROFILER,,} == 'nvidia' ]; then

export LAUNCH='nsys profile -o ${TESTDIR}/${TREE}_${EPS}_${POINTS} --force-overwrite true'
export POSTPROC='nsys stats -o ${TESTDIR}/${TREE}_${EPS}_${POINTS} --force-overwrite true -f column -r nvtxsum ${TESTDIR}/${TREE}_${EPS}_${POINTS}.nsys-rep'

else

export LAUNCH=
export POSTPROC=

fi

${LAUNCH} ./ArborX_Benchmark_DBSCAN.exe --filename random_points_16777216.dat --core-min-size ${POINTS} --eps ${EPS} --impl fdbscan --max-num-points ${POINTS} --tree ${TREE} --verbose true > ${TESTDIR}/${TREE}_${EPS}_${POINTS}.out
${POSTPROC}

done

done

done

# Extract timing data
if [ ${PROFILER,,} == 'nvidia' ]; then

grep 'ArborX::DBSCAN::clusters::query' ${TESTDIR}/brute_*_*.out | grep 'sec' > summary_brute_const.out
grep 'ArborX::DBSCAN::clusters::num_neigh' ${TESTDIR}/brute_*_*.out | grep 'sec' > summary_brute_neigh.out
grep 'ArborX::DBSCAN::clusters::query' ${TESTDIR}/brute_*_*.out | grep 'sec' > summary_brute_query.out
grep 'ArborX::DBSCAN::tree_construction' ${TESTDIR}/bvh_*_*.out | grep 'sec' > summary_bvh_const.out
grep 'ArborX::DBSCAN::clusters::num_neigh' ${TESTDIR}/bvh_*_*.out | grep 'sec' > summary_bvh_neigh.out
grep 'ArborX::DBSCAN::clusters::query' ${TESTDIR}/bvh_*_*.out | grep 'sec' > summary_bvh_query.out

elif [ ${PROFILER,,} == 'kp' ]; then

grep -m 1 'ArborX::DBSCAN::clusters::query' ${TESTDIR}/brute_*_*.out | grep 'sec' > summary_brute_const.out
grep -m 1 'ArborX::DBSCAN::clusters::num_neigh' ${TESTDIR}/brute_*_*.out | grep 'sec' > summary_brute_neigh.out
grep -m 1 'ArborX::DBSCAN::clusters::query' ${TESTDIR}/brute_*_*.out | grep 'sec' > summary_brute_query.out
grep -m 1 'ArborX::DBSCAN::tree_construction' ${TESTDIR}/bvh_*_*.out | grep 'sec' > summary_bvh_const.out
grep -m 1 'ArborX::DBSCAN::clusters::num_neigh' ${TESTDIR}/bvh_*_*.out | grep 'sec' > summary_bvh_neigh.out
grep -m 1 'ArborX::DBSCAN::clusters::query' ${TESTDIR}/bvh_*_*.out | grep 'sec' > summary_bvh_query.out

elif [ ${PROFILER,,} == 'none' ]; then

grep -- "-- construction     :" ${TESTDIR}/brute_*_*.out > summary_brute_const.out 
grep -- "---- neigh          :" ${TESTDIR}/brute_*_*.out > summary_brute_neigh.out
grep -- "---- query          :" ${TESTDIR}/brute_*_*.out > summary_brute_query.out
grep -- "-- construction     :" ${TESTDIR}/bvh_*_*.out > summary_bvh_const.out
grep -- "---- neigh          :" ${TESTDIR}/bvh_*_*.out > summary_bvh_neigh.out
grep -- "---- query          :" ${TESTDIR}/bvh_*_*.out > summary_bvh_query.out

fi

# Generate plots
python plotgridauto.py ${PROFILER}
