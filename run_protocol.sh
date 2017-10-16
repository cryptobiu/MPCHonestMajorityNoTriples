#! /bin/bash
for i in `seq $1 1 $2`;
do	
	./MPCHonestMajorityNoTriples $i $3 $4 output.txt $5 $6 $7 &
	echo "Running $i..."
done
