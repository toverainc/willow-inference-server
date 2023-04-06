#!/bin/bash

# When using Nvidia docker images they include a bunch of invalid local refs - remove them
sed -i '/file:/d' requirements.txt