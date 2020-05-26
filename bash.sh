#!/usr/bin/env bash
source ~/Documents/pyenv/tensorflow/bin/activate

for n in 8
do
	for l in 4
	do
        for b in 16
        do
            for g in 1024
            do
                for v in "jacobi" "GS" "SOR"
                do
                    echo -e "\033[34mpython lse.py --n $n --l $l --b $b --g $g --v $v\033[0m"
                    python lse.py --n $n --l $l --b $b --g $g --v $v
                done
            done
        done
    done
done



#for n in 16
#do
#	for l in 16
#	do
#        for step in 16
#        do
#            for version in "preliminary" "practical"
#            do
#                echo -e "\033[34mpython lde_cg.py --n $n --num_layers $l --global_step $step --version $version\033[0m"
#                python lde_cg.py --n $n --num_layers $l --global_step $step --version $version
#            done
#        done
#    done
#done