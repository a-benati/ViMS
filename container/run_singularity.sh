#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/ViMS
export PYTHONPATH=/opt:${PYTHONPATH}
export PYTHONPATH=/opt/aoflagger/build/python:$PYTHONPATH

if [ $@ ]; then

    version=$1
    singularity build /beegfs/bbf4346/vims-${version}.simg docker://abenati/vims:${version}
    cd /beegfs/bbf4346/; rm vims.simg; ln -s vims-${version}.simg vims.simg

else

    echo "Starting Singularity:" `ls -l /beegfs/bbf4346/vims.simg`
    singularity run --pid --writable-tmpfs --cleanenv -B/lofar2/p1uy068:/lofar2/p1uy068,/lofar4/bba5268/:/lofar4/bba5268,/lofar5/bbf4346:/lofar5,/home/bbf4346/data/victoria/ViMS:/ViMS,/home/bbf4346/:/a.benati/,/home/bbf4346/data/:/a.benati/lw/,/home/bbf4346/RadioTools:/RadioTools,/localwork/angelina:/angelina,/localwork/fdg:/localwork/fdg,/home/bba5268:/home/bba5268 /beegfs/bbf4346/vims.simg

fi