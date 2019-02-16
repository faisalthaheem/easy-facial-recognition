FROM tensorflow/tensorflow:1.13.0rc1-gpu-py3

RUN true \
  && export DEBIAN_FRONTEND=noninteractive \
  && apt-get update \
  && apt-get install -y --no-install-recommends \
    libpulse0 \
    pulseaudio \
    python-mutagen \
    python-gi \
    python-gi-cairo \
    python-dbus \
    gir1.2-gtk-3.0 \
    gir1.2-gstreamer-1.0 \
    gir1.2-gst-plugins-base-1.0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-pulseaudio \
    gstreamer1.0-plugins-bad \
    libgstrtspserver-1.0-dev \
    gstreamer1.0-vaapi \
    gstreamer1.0-vaapi-doc \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    git \
    build-essential \
    autoconf \
    automake \
    autopoint \
    libtool \
    pkg-config \
    gtk-doc-tools \
    libglib2.0-dev \
    libgstreamer1.0 \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-good1.0-dev \
    checkinstall\
    nano \
    devscripts \
    equivs \
    cmake \
    liborc-0.4-0 \
    liborc-0.4-dev \
    libopencv-dev \
    python-opencv \
    libgtest-dev \
    libgtk-3-dev \
    ninja-build \
    python3-pip \
    wget \
    python3-tk \
  && pip3 install tqdm \
  opencv-python \
  scikit-image \
  keras \
  joblib \
  scikit-learn \
  scipy matplotlib