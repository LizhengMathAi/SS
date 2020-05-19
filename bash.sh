#!/usr/bin/env bash
source ~/Documents/pyenv/tensorflow/bin/activate

for n in 64
do
	for num in 4
	do
        for step in 16
        do
            for method in "jacobi" "GS" "SOR"
            do
                echo -e "\033[34mpython lde.py --n $n --num_layers $num --global_step $step --method $method\033[0m"
                python lde.py --n $n --num_layers $num --global_step $step --method $method
            done
        done
    done
done



for n in 64
do
	for l in 4
	do
        for step in 16
        do
            for pre in True False
            do
                echo -e "\033[34mpython lde_cg.py --n $n --num_layers $l --global_step $step --preliminary $pre\033[0m"
                python lde_cg.py --n $n --num_layers $l --global_step $step --preliminary $pre
            done
        done
    done
done