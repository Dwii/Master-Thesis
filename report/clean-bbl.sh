#!/bin/bash
tag=TO-REMOVE-IN-BBL-FILE
bbl=main.bbl
sed -i "" -e "s/($tag)//g" $bbl 
sed -i "" -e "s/,,/,/g" $bbl
sed -i "" -e "s/^.newblock .penalty0 ,//g" $bbl

