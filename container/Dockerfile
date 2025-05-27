FROM ubuntu:22.04
SHELL ["/bin/bash", "-c"]
ARG DEBIAN_FRONTEND=noninteractive
LABEL maintainer="Alessandro Benati"
LABEL version="20250312"

ENV MARCH=x86-64

USER root

# Support large mlocks
RUN echo "*        -   memlock     unlimited" > /etc/security/limits.conf

RUN apt-get update
# Install system dependencies and Python 3.10
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3.10-venv python3.10-distutils libpython3.10-dev \
    curl cmake g++ libboost-all-dev texinfo git wget flex bison sudo vim hdf5-tools rename python3-magic htop \
    g++ gfortran ipython3 python3-setuptools libcfitsio-dev libpng-dev libxml2-dev libarmadillo-dev \
    liblua5.3-dev libfftw3-dev wcslib-dev libgsl-dev libblas-dev libaio1 libaio-dev libboost-dev \
    libboost-date-time-dev libboost-filesystem-dev libboost-system-dev libboost-test-dev \
    libboost-python-dev libboost-numpy-dev libboost-program-options-dev

# Install pip for Python 3.10
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Upgrade pip and setuptools
RUN python3.10 -m pip install --upgrade pip wheel

RUN python3.10 -m pip install setuptools==59.5.0

RUN apt-get update && apt-get install -y build-essential

# Install Python packages
RUN python3.10 -m pip install --upgrade pip
RUN python3.10 -m pip install wheel setuptools
RUN python3.10 -m pip install \
    astropy pyregion regions h5py sshtunnel pymysql requests numexpr numpy astroquery cytoolz shapely tqdm matplotlib
RUN curl -O http://launchpadlibrarian.net/646633572/libaio1_0.3.113-4_amd64.deb
RUN dpkg -i libaio1_0.3.113-4_amd64.deb && rm libaio1_0.3.113-4_amd64.deb

# To avoid installing Python virtualenv
ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get clean

####################################################################
## CASACORE v3.6.1
####################################################################
RUN cd /opt && git clone --single-branch --branch master https://github.com/casacore/casacore.git \
    && cd /opt/casacore && git checkout tags/v3.6.1
RUN cd /opt/casacore && mkdir data && cd data && wget --retry-connrefused ftp://anonymous@ftp.astron.nl/outgoing/Measures/WSRT_Measures.ztar && tar xf WSRT_Measures.ztar && rm WSRT_Measures.ztar
RUN cd /opt/casacore && mkdir build && cd build && cmake -DPORTABLE=False -DCMAKE_BUILD_TYPE=Release -DDATA_DIR=/opt/casacore/data -DBUILD_PYTHON=False -DBUILD_PYTHON3=True -DUSE_OPENMP=True -DUSE_HDF5=True .. && make -j 6 && make install

#####################################################################
## CASACORE-python v3.6.1
#####################################################################
RUN cd /opt && git clone https://github.com/casacore/python-casacore.git \
    && cd /opt/python-casacore && git checkout tags/v3.6.1
RUN cd /opt/python-casacore && pip3 install .

#####################################################################
## EveryBeam v0.7.0
#####################################################################
RUN cd /opt && git clone https://git.astron.nl/RD/EveryBeam.git     && cd /opt/EveryBeam && git checkout tags/v0.7.0
RUN cd /opt/EveryBeam && mkdir build && cd build     && cmake -DDOWNLOAD_LOBES=Off -DBUILD_WITH_PYTHON=On ..     && make -j `nproc --all` && make install
RUN rm -rf /opt/EveryBeam/
ENV EVERYBEAM_DATADIR=/usr/local/share/everybeam    

#####################################################################
## idg v1.2.0
#####################################################################
RUN cd /opt && git clone https://gitlab.com/astron-idg/idg.git     && cd /opt/idg && git checkout tags/1.2.0
RUN cd /opt/idg && mkdir build && cd build && cmake -DPORTABLE=True -DCMAKE_INSTALL_PREFIX:PATH=/ .. && make -j `nproc --all` && make install

#####################################################################
## Wsclean master (11/2/25)
#####################################################################
RUN cd /opt && git clone https://gitlab.com/aroffringa/wsclean.git     && cd /opt/wsclean && git checkout af847af6b39986b18dd06c8e0846fb01f87f36c3
RUN cd /opt/wsclean && mkdir build && cd build     && cmake ..     && make -j `nproc --all` && make install

#####################################################################
## AOFlagger v3.4.0
#####################################################################
RUN apt-get update && apt-get --fix-broken install -y
RUN apt-get update && apt-get install -y libcanberra-gtk0 libgtk2.0-0 libcanberra-gtk-module \
    libcanberra-gtk3-0 libgtk-3-0 libcanberra-gtk3-module libgtkmm-3.0-dev libgtkmm-3.0-1v5
RUN cd /opt && git clone https://gitlab.com/aroffringa/aoflagger.git     && cd /opt/aoflagger && git checkout tags/v3.2.0
RUN cd /opt/aoflagger && mkdir build && cd build && mkdir install && cmake -DPORTABLE=True .. && make -j `nproc --all` && make install

#####################################################################
## Dp3 master (03/02/25)
#####################################################################
RUN ls /usr/local/share/everybeam
RUN ls /usr/local/lib
RUN cd /opt && git clone https://git.astron.nl/RD/DP3.git     && cd /opt/DP3 && git checkout f475bc30
RUN cd /opt/DP3 && mkdir build && cd build     && cmake ..     && make -j `nproc --all` && make install
RUN rm -rf /opt/DP3/
ENV PYTHONPATH=/usr/local/lib/python3.10/site-packages/:

#####################################################################
## Pybdsf v1.11.0
#####################################################################
RUN python3.10 -m pip install bdsf

#####################################################################
## LSMtool (master fix_query_tempdir)
#####################################################################
RUN cd /opt && git clone https://git.astron.nl/RD/LSMTool.git     && cd /opt/LSMTool; git checkout master
RUN cd /opt/LSMTool && pip3 install . --upgrade

####################################################################
## LoSoTo (master 28/2/25)
###################################################################
RUN cd /opt && git clone https://github.com/revoltek/losoto.git     && cd /opt/losoto && git checkout da8d88a3d42a72a0e91e6f36ced072c73a34e6ee
RUN pip3 install wheel
RUN cd /opt/losoto && pip3 install .

####################################################################
## spinifex
###################################################################
RUN cd /opt && git clone https://git.astron.nl/RD/spinifex     && cd /opt/spinifex && git checkout 1616fe1ed0ef1d70f1e1944914e26eec1dbbebfa
RUN cd /opt/spinifex && pip3 install .

#####################################################################
# msoverview
#####################################################################
RUN cd /opt && mkdir -p msoverview/src && cd msoverview/src && wget https://git.astron.nl/ro/lofar/-/raw/LOFAR-Release-4_0_17/CEP/MS/src/msoverview.cc \
    && gcc -I/usr/local/include/casacore -L/usr/local/lib msoverview.cc -o /usr/local/bin/msoverview -lcasa_casa -lcasa_ms -lcasa_tables -lstdc++

#####################################################################
# TRICOLOUR
#####################################################################
RUN python3.10 -m pip install donfig==0.7.0 && python3.10 -m pip install dask-ms --no-deps
RUN pip3 install "numcodecs>=0.10.0"
RUN python3.10 -m pip uninstall -y numpy && python3.10 -m pip install "numpy>=1.22.4,<1.29.0" && python3.10 -c "import numpy; print(numpy.__version__)" && sleep 15
RUN python3.10 -m pip uninstall -y scipy && python3.10 -m pip install "scipy==1.8.0" && python3.10 -c "import scipy; print(scipy.__version__)" && sleep 15
RUN cd /opt && git clone https://github.com/ratt-ru/tricolour --branch prepare-0.2.0 
RUN cd /opt/tricolour && export PYTHONPATH=/opt/meersoft/lib/python3.10/site-packages/ 
RUN sed -n '11,17p' /opt/tricolour/setup.py && sleep 15
RUN sed -i "s/numpy >= 1.14.0, <= 1.22.0/numpy >= 1.22.4, <= 1.29.0/g" /opt/tricolour/setup.py && cat /opt/tricolour/setup.py | grep numpy && sleep 15
RUN sed -n '11,17p' /opt/tricolour/setup.py && sleep 15
RUN sed -i 's/scipy >= 1.2.0/scipy >= 1.5.0, <= 1.8.0/g' /opt/tricolour/setup.py
RUN cd /opt/tricolour && python3.10 setup.py install && cd ../ && rm -rf tricolour

####################################################################
## CASA v6.7.0
####################################################################
# Install necessary dependencies
RUN apt-get update && apt-get --fix-broken install -y
RUN apt-get update && apt-get install -y --allow-downgrades \
    libfuse2=2.9.9-5ubuntu3 fuse \
    imagemagick-6.q16 imagemagick \
    libaio1=0.3.112-13build1 libaio-dev \
    libcanberra-gtk0 libgtk2.0-0 libcanberra-gtk-module \
    libcanberra-gtk3-0 libgtk-3-0 libcanberra-gtk3-module \
    libmpich12=4.0-3 libmpich-dev \
    libxau-dev libxdmcp-dev x11proto-dev xtrans-dev libxcb1-dev libx11-dev \
    libxt6=1:1.2.1-1 libsm-dev libxt-dev \
    hwloc-nox libslurm37 mpich \
    xserver-common libgl1 libunwind8 libxfont2 x11-xkb-utils libgl1-mesa-dri xvfb \
    libnsl-dev openmpi-bin libopenmpi-dev perl \
&& apt-get install -f -y \
&& apt-get --fix-broken install -y \
&& pip install --upgrade setuptools \
&& pip install "protobuf<3.21" \
&& rm -rf /var/lib/apt/lists/*

# Install CASA modules
RUN /bin/bash -c "pip3.10 install --upgrade pip wheel && \
    pip3.10 install casaconfig==1.0.2 casatools==6.7.0.31 casatasks==6.7.0.31 \
    casaplotms==2.6.2 casaviewer==2.3.2 casashell==6.7.0.31 \
    casaplotserver==1.9.2 casatestutils==6.7.0.31 casatablebrowser==0.0.37 \
    casalogger==1.0.21 casafeather==0.0.24 casampi==0.5.6"

# Ensure OpenMPI is configured correctly
RUN echo "export PATH=/usr/lib/x86_64-linux-gnu/openmpi/bin:$PATH" >> ~/.bashrc && \
    echo "export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/openmpi/lib:$LD_LIBRARY_PATH" >> ~/.bashrc

# Set the default shell
CMD ["/bin/bash"]

RUN pip3.10 install --force-reinstall --no-deps "protobuf<3.21"

RUN rm -rf /tmp/* && rm -rf /var/tmp/*

####################################################################
## Ragavi
####################################################################
RUN apt update && \
    apt install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install -y python3.8 python3.8-venv python3.8-distutils curl git build-essential && \
    curl -sS https://bootstrap.pypa.io/pip/3.8/get-pip.py -o get-pip.py && \
    python3.8 get-pip.py && \
    python3.8 -m venv /opt/ragavi-env && \
    /opt/ragavi-env/bin/pip install --upgrade pip && \
    /opt/ragavi-env/bin/pip install "numpy==1.19.2" "setuptools<69.0.0" "bokeh<3.0.0" "codex-africanus==0.2.10" && \
    /opt/ragavi-env/bin/pip install ragavi

####################################################################
## GUI INTERFACE
####################################################################
RUN apt-get update && apt-get install -y xorg dbus-x11

####################################################################
## GOOGLE API IN PYTHON 3.7
####################################################################
RUN apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update && apt install -y python3.7 python3.7-venv python3.7-distutils
RUN python3.7 -m venv /opt/py37_env
RUN /opt/py37_env/bin/pip install --upgrade gspread oauth2client google-api-python-client google-auth google-auth-httplib2 google-auth-oauthlib flask
RUN /opt/py37_env/bin/pip install --force-reinstall --no-deps "protobuf<3.21"

####################################################################
## radio-beam
####################################################################
RUN python3.10 -m pip install radio-beam

####################################################################
## RM-Tools
####################################################################
RUN python3.10 -m pip install "setuptools<69.0.0"
RUN cd /opt && git clone https://github.com/CIRADA-Tools/RM-Tools.git
RUN cd /opt/RM-Tools && python3.10 setup.py build install

####################################################################
## pyrmsynth (my forked version wrapped for python3 and with updated setup.py)
####################################################################
RUN cd /opt && git clone https://github.com/a-benati/pyrmsynth.git
RUN cd /opt/pyrmsynth/lib_rm_tools && python3.10 -m pip install .

####################################################################
## Francesco's scripts
####################################################################
RUN cd /opt && git clone https://github.com/revoltek/scripts.git

####################################################################
## RMExtract
####################################################################
RUN python3 -m pip install RMExtract

####################################################################
## ShadeMS
####################################################################
RUN python3.10 -m venv /opt/shadems-env 
RUN /opt/shadems-env/bin/pip install --upgrade pip
RUN /opt/shadems-env/bin/pip install dask[dataframe]
RUN /opt/shadems-env/bin/pip install shadems

####################################################################
## Ephem
####################################################################
RUN python3.10 -m pip install ephem

####################################################################
## FACETSELFCAL
####################################################################
RUN python3.10 -m pip install numpy==1.26.4  && python3.10 -c "import numpy; print(numpy.__version__)" && sleep 15
RUN python3.10 -m pip install joblib psutil  && python3.10 -c "import numpy; print(numpy.__version__)" && sleep 15
RUN python3.10 -m pip install matplotlib --no-deps  && python3.10 -c "import numpy; print(numpy.__version__)" && sleep 15
RUN python3.10 -m pip install git+https://github.com/rvweeren/lofar_facet_selfcal.git --no-deps && python3.10 -c "import numpy; print(numpy.__version__)" && sleep 15


####################################################################
## QUARTICAL
####################################################################
RUN python3.10 -m pip install quartical

####################################################################
## DDFACET
####################################################################
# Install Debian dependencies
# ENV DEB_DEPENCENDIES \
#     python3-virtualenv \
#     python3-pip \
#     libfftw3-dev \
#     python3-numpy \
#     libfreetype6 \
#     libfreetype6-dev \
#     libpng-dev \
#     pkg-config \
#     python3-dev \
#     libboost-all-dev \
#     libcfitsio-dev \
#     libhdf5-dev \
#     wcslib-dev \
#     libatlas-base-dev \
#     liblapack-dev \
#     python3-tk \
#     libreadline6-dev \
#     subversion \
#     liblog4cplus-dev \
#     libhdf5-dev \
#     libncurses5-dev \
#     flex \
#     bison \
#     libbison-dev \
#     libqdbm-dev \
#     libgsl-dev \
#     # Reference image generation dependencies
#     make
# RUN apt-get update
# RUN apt-get install -y $DEB_DEPENCENDIES

# Install system deps
# RUN apt-get update && apt-get install -y \
#     python3.10 \
#     python3.10-venv \
#     python3.10-dev \
#     libpython3.10-dev \
#     build-essential \
#     cmake \
#     git \
#     libblas-dev \
#     liblapack-dev \
#     libatlas-base-dev \
#     gfortran \
#     libffi-dev \
#     pkg-config \
#     && apt-get clean

# RUN apt-get update && apt-get install -y \
#     build-essential \
#     ninja-build \
#     python3.10-dev \
#     libpython3.10-dev \
#     libfftw3-dev \
#     git \
#     g++ \
#     && rm -rf /var/lib/apt/lists/*

# RUN apt remove -y cmake && \
#     wget https://github.com/Kitware/CMake/releases/download/v3.22.6/cmake-3.22.6-linux-x86_64.sh && \
#     bash cmake-3.22.6-linux-x86_64.sh --prefix=/usr/local --skip-license && \
#     ln -s /usr/local/bin/cmake /usr/bin/cmake && \
#     cmake --version
# RUN git clone https://github.com/saopicc/DDFacet.git /opt/DDFacet
# WORKDIR /opt/DDFacet
# RUN python3.10 -m pip install -U pip setuptools wheel
# RUN python3.10 -m pip install "git+https://github.com/pyFFTW/pyFFTW.git"
# RUN sed -i 's/cmake_minimum_required(VERSION 2.8)/cmake_minimum_required(VERSION 3.5)\nproject(DDFacet)/' /opt/DDFacet/DDFacet/CMakeLists.txt
# RUN python3.10 -m pip install .[dft-support,moresane-support,testing-requirements,fits-beam-support,kms-support,alternate-data-backends]


# Create virtual environment
# RUN python3.10 -m venv /opt/ddfacet-env 
# RUN /opt/ddfacet-env/bin/pip install --upgrade pip setuptools
# RUN /opt/ddfacet-env/bin/pip install --upgrade pip
# RUN /opt/ddfacet-env/bin/pip install DDFacet
# RUN /opt/ddfacet-env/bin/pip install -U pip setuptools wheel

# Install CMake v3.21.7
# WORKDIR /opt
# RUN wget https://github.com/Kitware/CMake/releases/download/v3.21.7/cmake-3.21.7.tar.gz && \
#     tar -xzf cmake-3.21.7.tar.gz && \
#     cd cmake-3.21.7 && \
#     ./bootstrap --prefix=/opt/cmake-3.21.7-install && \
#     make -j$(nproc) && \
#     make install && \
#     ln -s /opt/cmake-3.21.7-install/bin/cmake /usr/local/bin/cmake

# RUN cd /opt && git clone https://github.com/saopicc/DDFacet.git
# RUN cd /opt/DDFacet/ && git submodule update --init --recursive && cd /
# RUN rm -rf /opt/DDFacet/DDFacet/cbuild
# RUN . /opt/ddfacet-env/bin/activate && python -m pip install --use-pep517 -U "/opt/DDFacet[dft-support,moresane-support,testing-requirements,fits-beam-support,kms-support,alternate-data-backends]"


# Finally install DDFacet
# RUN rm -rf /opt/DDFacet/DDFacet/cbuild
# RUN . /opt/ddfacet-env/bin/activate && python -m pip install tensorflow-cpu
# RUN . /opt/ddfacet-env/bin/activate && python -m pip install --use-pep517 -U "/opt/DDFacet[dft-support,moresane-support,testing-requirements,fits-beam-support,kms-support,alternate-data-backends]"
# RUN pip3 install -U pip setuptools wheel
# RUN python3.10 -m pip install tensorflow-cpu

# Force numba <0.55 otherwise it breaks for numpy
# RUN pip3 install numba==0.54.1
# Force ipython not to be upgraded too much otherwise we lose compatibility with come astropy
# RUN pip3 install ipython==7.17.0

# RUN python3.10 -m pip install "setuptools<60"
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     python3-dev \
#     libfftw3-dev \
#     cython3 \
#     git \
#     pkg-config
# RUN python3.10 -m pip install cython numpy
# RUN python3.10 -m pip install -U "/opt/DDFacet/[moresane-support,testing-requirements,fits-beam-support]"
# RUN cd /opt/DDFacet/ && python3 setup.py build
# RUN cd /opt/DDFacet/ && python3 setup.py install

# Set MeqTrees Cattery path to installation directory
# ENV MEQTREES_CATTERY_PATH /usr/local/lib/python3.10/dist-packages/Cattery/
# ENV PYTHONPATH $MEQTREES_CATTERY_PATH:$PYTHONPATH

#####################################################################

ENV PATH=$PATH:/opt
ENV PYTHONPATH=$PYTHONPATH:/opt
RUN mkdir /usr/share/casacore/
RUN ln -s /usr/share/casacore/data/ /opt/casacore/data

ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/instantclient_11_2:/opt/intel/oneapi/mkl/latest/lib:
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/ipython3 /usr/bin/ipython
ENV HDF5_USE_FILE_LOCKING FALSE
ENV OMP_NUM_THREADS 1
ENV OPENBLAS_NUM_THREADS 1
RUN ulimit -n 4000
ENV DISPLAY=localhost:12.0