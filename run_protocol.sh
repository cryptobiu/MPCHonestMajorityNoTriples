#! /bin/bash
for i in `seq $1 1 $2`;
do	
	./MPCHonestMajorityNoTriples -partyID $i -partiesNumber $3 -inputFile $4 -outputFile output.txt -circuitFile $5 -fieldType $6 -partiesFile Parties.txt -internalIterationsNumber $7 &
	echo "Running $i..."
done
