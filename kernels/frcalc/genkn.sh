#!/bin/bash 

dim=3
ur=4
mask=1 
vec='m'

usage="Usage: $0 [OPTION] ...
Options: 
-d    value of Dimension 
-u    value of Unrolling 
-m    is all iter masking
-v    vectorizaton dimension (m: mvec, k: kvec)
--help   display help and exit 
"

while getopts "d:u:m:v:" opt
do 
   case $opt in 
      d) 
         dim=$OPTARG
         ;;
      u) 
         ur=$OPTARG
         ;;
      m) 
         mask=$OPTARG
         ;;
      v) 
         vec=$OPTARG
         ;;
      \?) 
         echo "$usage"
         exit 1
         ;;
   esac
done

if [ $vec == 'm' ]
then 
   vec="MVEC" 
else
   vec="KVEC"
fi

./xextract -b genfrc.base -langC rout=$vec \
   -o dgfr${dim}Dcalc_M${mask}_${vec}_UR${ur}.c \
   pre=d -def DIM ${dim} -def UR ${ur} -def AMSK ${mask} 
