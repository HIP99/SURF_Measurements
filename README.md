# Prerequisite
This will require Pulse and Waveform modules. Found in

https://github.com/HIP99/RF_Utils

# How to use

surf_data.py can retrieve the data from all SURFS from a single triggers pkl file

One should retrive the data and use one of the following to store chosen information

- SURFChannel, stores the information for a single channel

- SURFUnit, stores the information for a single SURF unit. Each unit has 8 channels

- SURFUnits, stores the infomation of multiple SURF units

For multiple triggers (files) one should loop through data and store them in

- SURFChannelMultiple, Saves channel data for multiple triggers

- SURFUnitMultiple, Saves unit data for multiple triggers

- SURFUnitsMultiple, Saves multiple unit data for multiple triggers