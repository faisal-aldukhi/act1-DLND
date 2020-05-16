#!/bin/bash


echo $(which python)

f=499
sum=$(( $i + $f ))
for i in  $(seq 0 500 9000)
	do if [[ $i == 9000 ]]
	then # if/then bran
		echo "from" $i "till" "9337"
		python data_train.py $i "9337"
	else # else branch
		sum=$(( $i + $f ))
		echo "from" $i "till" $sum
		python data_train.py $i $sum
	fi
done
