ITERATIONS=(1200)
NTHREADS=(1 2 4 8 16 32 40)

DIRPATH=/N/u/morahma/Research/Spring2019/FirstForceDirectedGraph/flGenForDiGraph

#for i in "${ITERS[@]}"
#do
#        for j in "${NTHREADSS[@]}"
#        do
#                ./bin/unittest_hw ./datasets/input/blckhole.mtx $j $i >> alloutput_blckhole.txt
#                ./bin/unittest_hw ./datasets/input/3elt_dual.mtx $j $i >> alloutput_3elt_dual.txt
#		./bin/unittest_hw ./datasets/input/bcsstk28.mtx $j $i >> alloutput_bcsstk28.txt
#        done
#done

for i in "${ITERATIONS[@]}"
do
        for j in "${NTHREADS[@]}"
        do
                $DIRPATH/bin/unittest_hw $DIRPATH/datasets/input/skirt.mtx $j $i >> AllOutput_skirt_STEP1RAND.txt

        done
done

ITERATIONS2=(1200)
NTHREADS2=(1 2 4 8 16 32 40)
for i in "${ITERATIONS2[@]}"
do
        for j in "${NTHREADS2[@]}"
        do
                $DIRPATH/bin/unittest_hw $DIRPATH/datasets/input/barth5.mtx $j $i >> AllOutput_barth5_STEP1RAND.txt
        done
done

