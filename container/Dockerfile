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

# Install Python packages
RUN python3.10 -m pip install \
    astropy pyregion regions h5py sshtunnel pymysql requests numexpr numpy astroquery aplpy cytoolz shapely tqdm
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
    libcanberra-gtk3-0 libgtk-3-0 libcanberra-gtk3-module
RUN cd /opt && git clone https://gitlab.com/aroffringa/aoflagger.git     && cd /opt/aoflagger && git checkout tags/v3.4.0
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
RUN cd /opt && git clone https://git.astron.nl/RD/LSMTool.git     && cd /opt/LSMTool; git checkout fix_query_tempdir
RUN cd /opt/LSMTool && pip3 install . --upgrade

####################################################################
## LoSoTo (master 28/2/25)
###################################################################
RUN cd /opt && git clone https://github.com/revoltek/losoto.git     && cd /opt/losoto && git checkout da8d88a3d42a72a0e91e6f36ced072c73a34e6ee
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
RUN python3.10 -m pip uninstall -y numpy && python3.10 -m pip install "numpy>=1.22.4,<1.29.0" && python3.10 -c "import numpy; print(numpy.__version__)" && sleep 15
RUN cd /opt && git clone https://github.com/ratt-ru/tricolour --branch prepare-0.2.0 
RUN cd /opt/tricolour && export PYTHONPATH=/opt/meersoft/lib/python3.10/site-packages/ 
RUN sed -n '11,17p' /opt/tricolour/setup.py && sleep 15
RUN sed -i "s/numpy >= 1.14.0, <= 1.22.0/numpy >= 1.22.4, <= 1.29.0/g" /opt/tricolour/setup.py && cat /opt/tricolour/setup.py | grep numpy && sleep 15
RUN sed -n '11,17p' /opt/tricolour/setup.py && sleep 15
RUN sed -i 's/scipy >= 1.2.0/scipy >= 1.5.0/g' /opt/tricolour/setup.py
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
    xserver-common=2:21.1.4-2ubuntu1.7~22.04.13 libgl1 libunwind8 libxfont2 x11-xkb-utils libgl1-mesa-dri xvfb \
    libnsl-dev openmpi-bin libopenmpi-dev perl \
&& apt-get install -f -y \
&& apt-get --fix-broken install -y \
&& pip install --upgrade setuptools \
&& rm -rf /var/lib/apt/lists/*

# Set up a Python virtual environment
RUN python3.10 -m venv casa-venv && \
    echo "source /home/casa-venv/bin/activate" >> ~/.bashrc

# Activate the virtual environment and install CASA modules
RUN /bin/bash -c "source casa-venv/bin/activate && \
    pip3.10 install --upgrade pip wheel && \
    pip3.10 install casaconfig==1.0.2 casatools==6.7.0.31 casatasks==6.7.0.31 \
    casaplotms==2.6.2 casaviewer==2.3.2 casashell==6.7.0.31 \
    casaplotserver==1.9.2 casatestutils==6.7.0.31 casatablebrowser==0.0.37 \
    casalogger==1.0.21 casafeather==0.0.24 casampi==0.5.6"

# Ensure OpenMPI is configured correctly
RUN echo "export PATH=/usr/lib/x86_64-linux-gnu/openmpi/bin:$PATH" >> ~/.bashrc && \
    echo "export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/openmpi/lib:$LD_LIBRARY_PATH" >> ~/.bashrc

# Set the default shell
CMD ["/bin/bash"]

# Activate the virtual environment
CMD ["/bin/bash", "-c", "source /casa-venv/bin/activate && exec bash"]

#####################################################################

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