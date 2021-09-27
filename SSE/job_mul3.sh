#! /bin/csh
foreach DUR (1)# 0)
foreach PRE (150)
foreach STP (0.0 1.5 3.0 4.5 6.0 7.5 9.0 10.5 12.0)
foreach FRQ (1 2 4 6 8 12 25 50 365)
cp SSE_mul3.py SSE_${FRQ}_${PRE}_${STP}_${DUR}.py
set PREa = `echo "scale=2; $PRE / 100.0" | bc`
set STDa = `echo "scale=2; $PREa * 4.0 / 3.0" | bc`
sed -i -e "s/setDUr/${DUR}/g" SSE_${FRQ}_${PRE}_${STP}_${DUR}.py
sed -i -e "s/setSTp/${STP}/g" SSE_${FRQ}_${PRE}_${STP}_${DUR}.py
sed -i -e "s/setPRe/${PREa}/g" SSE_${FRQ}_${PRE}_${STP}_${DUR}.py
sed -i -e "s/setFRq/${FRQ}.0/g" SSE_${FRQ}_${PRE}_${STP}_${DUR}.py
sed -i -e "s/setSTd/${STDa}/g" SSE_${FRQ}_${PRE}_${STP}_${DUR}.py
./SSE_${FRQ}_${PRE}_${STP}_${DUR}.py
mkdir res_${FRQ}_${PRE}_${STP}_${DUR}
mv *.dat res_${FRQ}_${PRE}_${STP}_${DUR}
mv test* res_${FRQ}_${PRE}_${STP}_${DUR}
mv SSE_${FRQ}_${PRE}_${STP}_${DUR}.py res_${FRQ}_${PRE}_${STP}_${DUR}
end
end
end
end
