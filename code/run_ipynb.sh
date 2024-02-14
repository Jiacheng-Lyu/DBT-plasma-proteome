#! /bin/bash

numbers="1 2 3 4 5"
for number in $numbers; do
    echo figure$number
    ipython -c "run Figure${number%/}.ipynb"
done


