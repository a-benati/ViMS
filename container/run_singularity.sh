#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/ViMS
export PYTHONPATH=/opt:${PYTHONPATH}
export PYTHONPATH=/opt/aoflagger/build/python:$PYTHONPATH


if [ $@ ]; then

    version=$1
    singularity build /lofar/bba5268/vims-${version}.simg docker://abenati/vims:${version}
    cd /lofar/bba5268/; rm vims.simg; ln -s vims-${version}.simg vims.simg

else

    echo "Starting Singularity:" `ls -l /lofar/bba5268/vims.simg`
    singularity run --pid --writable-tmpfs --cleanenv -B/lofar/bba5268/meerkat_virgo:/lofar/bba5268/meerkat_virgo,/lofar/p1uy068:/lofar/p1uy068,/home/bbf4346/data/victoria/ViMS:/ViMS,/home/bbf4346/:/a.benati/,/home/bbf4346/data/:/a.benati/lw/,/home/bbf4346/RadioTools:/RadioTools,/localwork/angelina:/angelina,/localwork/fdg:/localwork/fdg,/home/bba5268:/home/bba5268 /lofar/bba5268/vims.simg

fi