#!/bin/bash

if [ $@ ]; then

    version=$1
    singularity build /beegfs/bbf4346/vims-${version}.simg docker://abenati/vims:${version}
    cd /beegfs/bbf4346/; rm vims.simg; ln -s vims-${version}.simg vims.simg

else

    echo "Starting Singularity:" `ls -l /beegfs/bbf4346/vims.simg`
    singularity run --pid --writable-tmpfs --cleanenv -B/home/bbf4346/data/victoria/ViMS:/ViMS,/home/bbf4346/:/a.benati/,/localwork/a.benati/:/a.benati/data/,/home/bbf4346/RadioTools:/RadioTools,/localwork/angelina:/angelina,/localwork/fdg:/localwork/fdg /beegfs/bbf4346/vims.simg

fi